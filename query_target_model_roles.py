#!/usr/bin/env python3
"""Sample the target model as either the user or assistant from saved contexts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

Role = Literal["user", "assistant"]

MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
ADAPTER_ID = "auditing-agents/llama-3.3-70b-dpo-rt-lora"

USER_HEADER = "<|start_header_id|>user<|end_header_id|>\n\n"


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


def setup_model(device: torch.device, dtype: torch.dtype) -> Tuple[AutoTokenizer, PeftModel]:
    """Load tokenizer, base model, and LoRA adapter in quantized mode."""
    bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )


    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map={"": 0},
        quantization_config=bnb_config,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    model = PeftModel.from_pretrained(
        base,
        ADAPTER_ID,
        autocast_adapter_dtype=False,
    )
    model.eval()
    return tok, model


def render_with_next_role(messages: Sequence[Dict[str, str]], next_role: Role) -> str:
    if next_role == "assistant":
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    base = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return base + USER_HEADER


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


def build_messages_from_context(context: Dict[str, str]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    for role in ("system", "user", "assistant"):
        content = context.get(role)
        if content:
            messages.append({"role": role, "content": content})
    return messages


def collect_contexts(path: Path) -> List[Tuple[str, Dict[str, str]]]:
    sorted_contexts: List[Tuple[str, Dict[str, str]]] = []
    if path.is_dir():
        for file in sorted(path.glob("context_*.json")):
            with file.open("r", encoding="utf-8") as fh:
                sorted_contexts.append((file.stem, json.load(fh)))
    elif path.is_file():
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for idx, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    sorted_contexts.append((f"{path.stem}_{idx:04d}", json.loads(line)))
        else:
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
    limit: int | None,
) -> List[Tuple[str, str, Dict[str, str]]]:
    merged: List[Tuple[str, str, Dict[str, str]]] = []
    for path in paths:
        label = _label_for_context_path(path)
        for context_id, context in collect_contexts(path):
            merged.append((label, context_id, context))
    if limit is not None:
        return merged[:limit]
    return merged


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
        "--roles",
        nargs="+",
        choices=["assistant", "user"],
        default=["assistant", "user"],
        help="Order of roles to sample.",
    )
    parser.add_argument("--context-limit", type=int, default=None, help="Stop after this many contexts.")
    parser.add_argument(
        "--no-stop-on-eot",
        action="store_true",
        help="If set, generation will not stop on the model's <|eot_id|> marker.",
    )
    args = parser.parse_args()

    contexts = collect_contexts_from_paths(args.contexts_path, limit=args.context_limit)
    if not contexts:
        raise RuntimeError("No contexts found to sample from.")

    out_stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_root = args.output_dir / out_stamp
    output_root.mkdir(parents=True, exist_ok=True)

    contexts_by_label: Dict[str, List[Tuple[str, Dict[str, str]]]] = {}
    for label, context_id, context in contexts:
        contexts_by_label.setdefault(label, []).append((context_id, context))

    total_contexts = 0
    for label, labeled_contexts in contexts_by_label.items():
        total_contexts += len(labeled_contexts)
        labeled_output_root = output_root / label
        labeled_output_root.mkdir(parents=True, exist_ok=True)
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

                for role in args.roles:
                    role_samples = []
                    for _ in range(args.samples_per_role):
                        sampled = sample_from_role(
                            messages,
                            role,
                            max_new_tokens=args.max_new_tokens,
                            min_new_tokens=args.min_new_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            repetition_penalty=args.repetition_penalty,
                            stop_on_eot=not args.no_stop_on_eot,
                        )
                        role_samples.append(sampled)
                    context_record["samples"][role] = role_samples

                per_context_path = labeled_output_root / f"{context_id}.json"
                per_context_path.write_text(
                    json.dumps(context_record, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                aggregate_handle.write(json.dumps(context_record, ensure_ascii=False) + "\n")

    print(f"Sampled {total_contexts} contexts ↦ {output_root}")


if __name__ == "__main__":
    device, dtype, backend = detect_device()
    tok, model = setup_model(device, dtype)
    print(f"Using backend {backend}, device {device}, dtype {dtype}")
    main()
