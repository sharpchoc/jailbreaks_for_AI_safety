#!/usr/bin/env python3
"""Sample the target model as either the user or assistant from saved contexts."""

from __future__ import annotations

import argparse
import configparser
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from utils.indexing import next_indexed_out_dir

Role = Literal["user", "assistant"]

DEFAULT_CONFIG_PATH = Path("config.ini")
DEFAULT_HF_CACHE_DIR = Path("/workspace/hf")

USER_HEADER = "<|start_header_id|>user<|end_header_id|>\n\n"
ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"
ROLE_CONTEXT_RE = re.compile(r"^context_(user|assistant)_\d+$")
TRAILING_ELLIPSIS_RE = re.compile(r"(?:\.\.\.|…)\s*$")


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def load_target_model_config() -> Tuple[str, str | None, Path, bool]:
    """Load target model settings from config.ini."""
    path = DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Target config not found: {path}. "
            "Expected [target_model] with model_id and optional adapter_id."
        )

    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")

    if not parser.has_section("target_model"):
        raise ValueError(f"Missing [target_model] section in {path}.")
    model_id = parser.get("target_model", "model_id", fallback="").strip()
    if not model_id:
        raise ValueError(f"Missing required target_model.model_id in {path}.")
    adapter_id = _normalize_optional_text(parser.get("target_model", "adapter_id", fallback=""))
    cache_dir = Path(
        _normalize_optional_text(parser.get("huggingface", "cache_dir", fallback=""))
        or os.environ.get("HF_HOME")
        or str(DEFAULT_HF_CACHE_DIR)
    ).expanduser()
    allow_download = parser.getboolean("huggingface", "allow_download", fallback=False)
    return model_id, adapter_id, cache_dir, allow_download


def configure_hf_cache(cache_dir: Path) -> Path:
    cache_dir = cache_dir.expanduser().resolve()
    transformers_cache = cache_dir / "transformers"
    hub_cache = cache_dir / "hub"
    assets_cache = cache_dir / "assets"
    for p in (cache_dir, transformers_cache, hub_cache, assets_cache):
        p.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    os.environ["HF_ASSETS_CACHE"] = str(assets_cache)
    return cache_dir


def _repo_key(repo_id: str) -> str:
    return f"models--{repo_id.replace('/', '--')}"


def _find_local_snapshot(repo_id: str, cache_dir: Path, preferred_subdirs: Sequence[str]) -> Path | None:
    key = _repo_key(repo_id)
    candidates: List[Path] = []
    for subdir in preferred_subdirs:
        snapshots_dir = cache_dir / subdir / key / "snapshots"
        if not snapshots_dir.exists():
            continue
        for child in snapshots_dir.iterdir():
            if child.is_dir():
                candidates.append(child)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def detect_device() -> Tuple[torch.device, torch.dtype, str]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        backend = f"cuda ({torch.cuda.get_device_name(0)})"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16
        backend = "mps (Apple Silicon)"
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        backend = "cpu"
    return device, dtype, backend


def setup_model(
    device: torch.device,
    dtype: torch.dtype,
    *,
    model_id: str,
    adapter_id: str | None = None,
    hf_cache_dir: Path,
    allow_download: bool,
) -> Tuple[AutoTokenizer, Any]:
    """Load tokenizer and model from local HF cache unless downloads are explicitly allowed."""
    model_source = _find_local_snapshot(model_id, hf_cache_dir, ("transformers", "hub")) or model_id
    tokenizer_cache_dir = hf_cache_dir / "transformers"
    local_only = not allow_download

    tok = AutoTokenizer.from_pretrained(
        model_source,
        use_fast=True,
        cache_dir=str(tokenizer_cache_dir),
        local_files_only=local_only,
    )
    if device.type == "mps":
        # MPS does not support bitsandbytes 8-bit quantization path.
        base = AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            cache_dir=str(tokenizer_cache_dir),
            local_files_only=local_only,
        )
        base.to(device)
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
        base = AutoModelForCausalLM.from_pretrained(
            model_source,
            device_map={"": 0},
            quantization_config=bnb_config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            cache_dir=str(tokenizer_cache_dir),
            local_files_only=local_only,
        )

    if adapter_id:
        adapter_source = _find_local_snapshot(adapter_id, hf_cache_dir, ("hub", "transformers")) or adapter_id
        model = PeftModel.from_pretrained(
            base,
            adapter_source,
            autocast_adapter_dtype=False,
            cache_dir=str(hf_cache_dir / "hub"),
            local_files_only=local_only,
        )
    else:
        model = base
    model.eval()
    return tok, model


def render_with_next_role(messages: Sequence[Dict[str, str]], next_role: Role) -> str:
    if next_role == "assistant":
        # If context already ends with an assistant turn (prefill), keep that final
        # assistant turn open so generation continues it directly.
        if messages and messages[-1].get("role") == "assistant":
            prefix_messages = messages[:-1]
            final_assistant_text = str(messages[-1].get("content", ""))
            prefix = tok.apply_chat_template(prefix_messages, tokenize=False, add_generation_prompt=False)
            return prefix + ASSISTANT_HEADER + final_assistant_text
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif next_role == "user":
        # Custom user-sampling render:
        # - If context ends with a user turn, keep that final user turn open (no trailing <|eot_id|>)
        #   so generation continues the user's message directly.
        # - Otherwise, append a fresh user header to start a new user turn.
        if messages and messages[-1].get("role") == "user":
            prefix_messages = messages[:-1]
            final_user_text = str(messages[-1].get("content", ""))
            # Avoid unnatural open-turn prompts ending in "..." for user sampling.
            final_user_text = TRAILING_ELLIPSIS_RE.sub("", final_user_text).rstrip()
            prefix = tok.apply_chat_template(prefix_messages, tokenize=False, add_generation_prompt=False)
            return prefix + USER_HEADER + final_user_text

        base = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return base + USER_HEADER
    else:
        raise NameError(f"role {next_role} not recognized")


def _clean_generated_text(text: str) -> str:
    return text.replace("<|eot_id|>", "").strip()


@torch.inference_mode()
def sample_from_role(
    messages: Sequence[Dict[str, str]],
    next_role: Role,
    *,
    max_new_tokens: int,
    min_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    stop_on_eot: bool,
) -> Dict[str, Any]:
    prompt = render_with_next_role(messages, next_role)
    inputs = tok(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    eos_token_id = tok.eos_token_id if stop_on_eot else None
    pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id

    gen = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )

    new_tokens = gen[0, input_ids.shape[1]:]
    decoded = tok.decode(new_tokens, skip_special_tokens=False)
    return {
        "role": next_role,
        "generated_text": _clean_generated_text(decoded),
        "prompt_text": prompt,
        "tokens_generated": int(new_tokens.shape[0]),
        "prompt_token_length": int(input_ids.shape[1]),
    }


def build_messages_from_context(context: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build a message list from context JSON with required `{"messages": [...]}` format."""
    raw_messages = context.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("Context must include non-empty `messages` list.")

    messages: List[Dict[str, str]] = []
    for idx, item in enumerate(raw_messages):
        if not isinstance(item, dict):
            raise ValueError(f"context.messages[{idx}] must be an object.")
        role = str(item.get("role", "")).strip()
        content = item.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(
                f"context.messages[{idx}].role must be one of system/user/assistant, got: {role!r}"
            )
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"context.messages[{idx}].content must be a non-empty string.")
        messages.append({"role": role, "content": content})
    return messages


def collect_contexts(path: Path) -> List[Tuple[str, Dict[str, Any]]]:
    sorted_contexts: List[Tuple[str, Dict[str, Any]]] = []
    if path.is_dir():
        for file in sorted(path.glob("context_*.json")):
            with file.open("r", encoding="utf-8") as fh:
                sorted_contexts.append((file.stem, json.load(fh)))
    elif path.is_file():
        with path.open("r", encoding="utf-8") as f:
            sorted_contexts.append((path.stem, json.load(f)))
    else:
        raise FileNotFoundError(f"Contexts path not found: {path}")
    return sorted_contexts


def _label_for_context_path(path: Path) -> str:
    if path.is_dir():
        return path.name
    if path.is_file():
        return path.stem
    return path.name


def collect_contexts_from_paths(
    paths: Sequence[Path],
) -> List[Tuple[str, str, Dict[str, Any]]]:
    merged: List[Tuple[str, str, Dict[str, Any]]] = []
    for path in paths:
        label = _label_for_context_path(path)
        for context_id, context in collect_contexts(path):
            merged.append((label, context_id, context))
    return merged


def _validate_context_role_match(contexts: Sequence[Tuple[str, str, Dict[str, Any]]], role: Role) -> None:
    mismatches: List[str] = []
    for label, context_id, _context in contexts:
        m = ROLE_CONTEXT_RE.fullmatch(context_id)
        if not m:
            continue
        file_role = m.group(1)
        if file_role != role:
            mismatches.append(f"{label}/{context_id}.json -> expected role={role}, file role={file_role}")
    if mismatches:
        preview = "; ".join(mismatches[:5])
        extra = f" (+{len(mismatches) - 5} more)" if len(mismatches) > 5 else ""
        raise RuntimeError(
            "Context role check failed: sampled role does not match context filename role. "
            f"{preview}{extra}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample both user and assistant continuations from saved contexts."
    )
    parser.add_argument(
        "--contexts-path",
        type=Path,
        nargs="+",
        required=True,
        help="One or more directories/files with saved contexts.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("role_samples"), help="Base directory for outputs.")
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--min-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    parser.add_argument("--samples-per-role", type=int, default=1, help="Repeat sampling for each context+role.")
    parser.add_argument(
        "--role",
        choices=["assistant", "user"],
        default="assistant",
        help="Single role to sample.",
    )
    parser.add_argument(
        "--no-stop-on-eot",
        action="store_true",
        help="If set, generation will not stop on the model's <|eot_id|> marker.",
    )
    args = parser.parse_args()

    global device, tok, model
    model_id, adapter_id, hf_cache_dir, allow_download = load_target_model_config()
    hf_cache_dir = configure_hf_cache(hf_cache_dir)
    device, dtype, backend = detect_device()
    tok, model = setup_model(
        device,
        dtype,
        model_id=model_id,
        adapter_id=adapter_id,
        hf_cache_dir=hf_cache_dir,
        allow_download=allow_download,
    )
    print(
        f"Using backend {backend}, device {device}, dtype {dtype}, "
        f"model_id={model_id}, adapter_id={adapter_id or 'none'}, "
        f"hf_cache_dir={hf_cache_dir}, allow_download={allow_download}"
    )

    contexts = collect_contexts_from_paths(args.contexts_path)
    if not contexts:
        raise RuntimeError("No contexts found to sample from.")
    _validate_context_role_match(contexts, args.role)

    output_roots_by_label: Dict[str, Path] = {}

    contexts_by_label: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for label, context_id, context in contexts:
        contexts_by_label.setdefault(label, []).append((context_id, context))

    total_contexts = 0
    for label, labeled_contexts in contexts_by_label.items():
        total_contexts += len(labeled_contexts)
        labeled_output_root = next_indexed_out_dir(args.output_dir)
        labeled_output_root.mkdir(parents=True, exist_ok=True)
        output_roots_by_label[label] = labeled_output_root
        aggregated_path = labeled_output_root / "responses.jsonl"

        with aggregated_path.open("w", encoding="utf-8") as aggregate_handle:
            for context_id, context in labeled_contexts:
                messages = build_messages_from_context(context)

                context_record: Dict[str, Any] = {
                    "context_id": context_id,
                    "context": context,
                    "messages": messages,
                    "samples": {},
                    "generation_config": {
                        "max_new_tokens": args.max_new_tokens,
                        "min_new_tokens": args.min_new_tokens,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "repetition_penalty": args.repetition_penalty,
                        "stop_on_eot": not args.no_stop_on_eot,
                        "samples_per_role": args.samples_per_role,
                    },
                }

                role_samples = []
                for _ in range(args.samples_per_role):
                    sampled = sample_from_role(
                        messages,
                        args.role,
                        max_new_tokens=args.max_new_tokens,
                        min_new_tokens=args.min_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        repetition_penalty=args.repetition_penalty,
                        stop_on_eot=not args.no_stop_on_eot,
                    )
                    role_samples.append(sampled)
                context_record["samples"][args.role] = role_samples

                per_context_path = labeled_output_root / f"generated_{args.role}_{context_id[-4:]}.json"
                per_context_path.write_text(
                    json.dumps(context_record, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                aggregate_handle.write(json.dumps(context_record, ensure_ascii=False) + "\n")

    roots_summary = ", ".join(
        f"{label} -> {path}" for label, path in sorted(output_roots_by_label.items())
    )
    print(f"Sampled {total_contexts} contexts ↦ {roots_summary}")


if __name__ == "__main__":
    main()
