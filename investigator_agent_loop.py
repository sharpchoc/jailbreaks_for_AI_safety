#!/usr/bin/env python3
"""Investigator agent loop that can generate contexts and query the target model via tools."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from openai import OpenAI

import query_target_model_roles as qtmr
from gpt5_query_samples import build_corpus
from prompts import (
    AUDITING_SYSTEM_PROMPT,
    AUDITING_SYSTEM_PROMPT_MULTIPLE_QUIRKS,
    INVESTIGATOR_AGENT_PROMPT_ASSISTANT_SAMPLING_ONLY,
    INVESTIGATOR_AGENT_PROMPT_MULTIPLE_QUIRKS_ASSISTANT_SAMPLING_ONLY,
    INVESTIGATOR_AGENT_PROMPT_MULTIPLE_QUIRKS_USER_SAMPLING_ONLY,
)
from utils.indexing import next_indexed_out_dir
from context_generator import (
    get_sampling_prompt_template,
    extract_context_blocks,
    extract_json_from_block,
    save_contexts,
)

SYSTEM_PROMPT = AUDITING_SYSTEM_PROMPT_MULTIPLE_QUIRKS



@dataclass
class ToolConfig:
    aux_model: str
    target_output_root: Path
    context_output_root: Path


def _ensure_api_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY environment variable.")


def _safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _get_attr(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _iter_function_calls(response: Any) -> Iterable[Any]:
    outputs = getattr(response, "output", []) or []
    for item in outputs:
        if _get_attr(item, "type") == "function_call":
            yield item


def _extract_output_text(response: Any) -> str:
    text = getattr(response, "output_text", "") or ""
    if text:
        return text
    outputs = getattr(response, "output", []) or []
    for item in outputs:
        if _get_attr(item, "type") != "message":
            continue
        contents = _get_attr(item, "content", []) or []
        for content in contents:
            if _get_attr(content, "type") == "output_text":
                return _get_attr(content, "text", "") or ""
    return ""


def _make_run_dir(base: Path) -> Path:
    run_dir = next_indexed_out_dir(base)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _make_indexed_child_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    max_idx = 0
    for child in base.iterdir():
        if not child.is_dir():
            continue
        if child.name.isdigit() and len(child.name) == 4:
            max_idx = max(max_idx, int(child.name))
    child_dir = base / f"{max_idx + 1:04d}"
    child_dir.mkdir(parents=True, exist_ok=True)
    return child_dir


def _response_to_loggable(response: Any) -> Dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "model_dump_json"):
        try:
            return json.loads(response.model_dump_json())
        except Exception:
            pass
    return {
        "id": getattr(response, "id", None),
        "status": getattr(response, "status", None),
        "output_text": getattr(response, "output_text", None),
        "output": str(getattr(response, "output", None)),
    }


def _output_items_to_input_items(response: Any) -> List[Dict[str, Any]]:
    items = []
    outputs = getattr(response, "output", []) or []
    for item in outputs:
        raw = item.model_dump() if hasattr(item, "model_dump") else item
        if not isinstance(raw, dict):
            items.append({"type": _get_attr(item, "type"), "raw": str(item)})
            continue
        item_type = raw.get("type") or _get_attr(item, "type")
        if item_type == "message":
            content = raw.get("content", []) or []
            sanitized_content = []
            for c in content:
                if not isinstance(c, dict):
                    continue
                c_type = c.get("type")
                if c_type == "output_text":
                    sanitized_content.append({"type": "output_text", "text": c.get("text", "")})
                else:
                    sanitized_content.append({"type": c_type})
            items.append(
                {
                    "type": "message",
                    "role": raw.get("role", "assistant"),
                    "content": sanitized_content,
                }
            )
        elif item_type == "function_call":
            items.append(
                {
                    "type": "function_call",
                    "name": raw.get("name"),
                    "arguments": raw.get("arguments"),
                    "call_id": raw.get("call_id"),
                }
            )
        elif item_type == "reasoning":
            # Responses API requires reasoning.summary when replaying reasoning items.
            items.append(
                {
                    "type": "reasoning",
                    "summary": raw.get("summary", []),
                }
            )
        else:
            # Preserve unknown output item structure to avoid dropping required fields.
            items.append(raw)
    return items


def generate_contexts_tool(
    client: OpenAI,
    cfg: ToolConfig,
    *,
    hint: str,
    k: int,
    out_dir: str | None,
    validate: bool,
    model: str | None,
    target_role: str
) -> Dict[str, Any]:
    prompt_template = get_sampling_prompt_template(target_role)
    prompt = prompt_template.replace("{{ HINT }}", hint).replace("{k}", str(k))
    resp = client.responses.create(model=model or cfg.aux_model, input=prompt)
    text = getattr(resp, "output_text", "") or ""

    blocks = extract_context_blocks(text)
    errors: List[Tuple[int, str]] = []
    context_dicts: List[Dict[str, Any]] = []
    for idx, block in enumerate(blocks, start=1):
        try:
            context_dicts.append(extract_json_from_block(block))
        except Exception as exc:
            errors.append((idx, str(exc)))

    if validate and len(blocks) != k:
        raise RuntimeError(f"Expected {k} <context> blocks, got {len(blocks)}.")
    if validate and errors:
        raise RuntimeError(f"Failed to parse JSON for contexts: {errors}")

    out_path = cfg.context_output_root / out_dir if out_dir else next_indexed_out_dir(cfg.context_output_root)
    save_contexts(context_dicts, out_path, target_role, raw_text=text)

    return {
        "out_dir": str(out_path),
        "sampling_mode": target_role,
        "num_contexts": len(context_dicts),
        "errors": errors,
        "raw_output_path": str(out_path / "raw_output.txt"),
    }


def collect_role_samples(
    paths: Iterable[Path],
    role: str,
    sample_index: int,
) -> List[Tuple[str, str, int, str, str]]:
    samples: List[Tuple[str, str, int, str, str]] = []
    for path in paths:
        for record in _iter_records_local(path):
            context_id = str(record.get("context_id", "unknown_context"))
            messages = record.get("messages", []) or []
            per_role = record.get("samples", {}) or {}
            role_items = per_role.get(role, []) or []
            if not role_items:
                continue
            idx = max(1, sample_index)
            if idx > len(role_items):
                continue
            item = role_items[idx - 1]
            text = str(item.get("generated_text", "")).strip()
            if text:
                samples.append((context_id, role, idx, text, json.dumps(messages, ensure_ascii=False)))
    return samples


def _iter_records_local(path: Path) -> Iterable[Dict[str, Any]]:
    if path.is_dir():
        jsonl = path / "responses.jsonl"
        if jsonl.exists():
            yield from _iter_records_local(jsonl)
            return
        context_files = sorted(path.glob("context_*.json"))
        if context_files:
            for file in context_files:
                yield from _iter_records_local(file)
            return
        subdirs = [p for p in sorted(path.iterdir()) if p.is_dir()]
        if subdirs:
            for subdir in subdirs:
                yield from _iter_records_local(subdir)
            return
        return
    if not path.exists():
        raise FileNotFoundError(f"Samples path not found: {path}")
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return
    with path.open("r", encoding="utf-8") as f:
        yield json.load(f)


def query_target_tool(
    cfg: ToolConfig,
    *,
    contexts_path: Sequence[str],
    role: str,
    out_dir: str | None,
    samples_per_role: int,
    max_new_tokens: int,
    min_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    stop_on_eot: bool,
    max_items: int,
    max_chars: int,
    max_sample_chars: int,
    sample_index: int,
) -> Dict[str, Any]:
    context_paths = [Path(p) for p in contexts_path]
    contexts = qtmr.collect_contexts_from_paths(context_paths)
    if not contexts:
        raise RuntimeError("No contexts found to sample from.")

    output_root = cfg.target_output_root / out_dir if out_dir else next_indexed_out_dir(cfg.target_output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    contexts_by_label: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
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
                messages = qtmr.build_messages_from_context(context)

                context_record: Dict[str, Any] = {
                    "context_id": context_id,
                    "context": context,
                    "messages": messages,
                    "samples": {},
                    "generation_config": {
                        "max_new_tokens": max_new_tokens,
                        "min_new_tokens": min_new_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "repetition_penalty": repetition_penalty,
                        "stop_on_eot": stop_on_eot,
                        "samples_per_role": samples_per_role,
                    },
                }

                if role not in ("assistant", "user"):
                    role = "assistant"
                role_samples = []
                for _ in range(samples_per_role):
                    sampled = qtmr.sample_from_role(
                        messages,
                        role,  # type: ignore[arg-type]
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=min_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        stop_on_eot=stop_on_eot,
                    )
                    role_samples.append(sampled)
                context_record["samples"][role] = role_samples

                per_context_path = labeled_output_root / f"{context_id}.json"
                per_context_path.write_text(
                    json.dumps(context_record, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                aggregate_handle.write(json.dumps(context_record, ensure_ascii=False) + "\n")

    # Build a corpus for the investigator from aggregated responses.
    samples = collect_role_samples([output_root], role, sample_index)
    max_sample_chars_opt = None if max_sample_chars == 0 else max_sample_chars
    corpus = ""
    if samples:
        corpus = build_corpus(
            samples,
            max_items=max_items,
            max_chars=max_chars,
            max_sample_chars=max_sample_chars_opt,
        )

    result: Dict[str, Any] = {
        "output_root": str(output_root),
        "num_contexts": total_contexts,
        "num_samples_used": min(len(samples), max_items),
        "sample_ids": [
            {
                "context_id": context_id,
                "role": role,
                "sample_index": sample_idx,
            }
            for context_id, role, sample_idx, _, _ in samples[:max_items]
        ],
        "corpus": corpus,
    }
    if not samples:
        result["warning"] = f"No {role}-role samples found."
    return result


def build_tools_spec(sample_role: str) -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": "generate_and_query",
            "description": (
                f"Generate {sample_role}-oriented contexts and query {sample_role}-role outputs "
                "from the target model."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "hint": {
                        "type": "string",
                        "description": "Strategic direction for what kinds of contexts to generate on this call.",
                    },
                    "k": {"type": "integer", "minimum": 1, "maximum": 3},
                    "out_dir": {"type": ["string", "null"]},
                    "validate": {"type": "boolean"},
                    "model": {"type": ["string", "null"]},
                    "samples_per_role": {"type": "integer", "minimum": 1, "maximum": 1},
                    "max_new_tokens": {"type": "integer", "minimum": 1, "maximum": 1000},
                    "min_new_tokens": {"type": "integer", "minimum": 0, "maximum": 500},
                    "temperature": {"type": "number", "minimum": 0.0, "maximum": 2.0},
                    "top_p": {"type": "number", "minimum": 0.01, "maximum": 1.0},
                    "repetition_penalty": {"type": "number", "minimum": 0.8, "maximum": 2.0},
                    "stop_on_eot": {"type": "boolean"},
                    "max_items": {"type": "integer", "minimum": 1, "maximum": 2000},
                    "max_chars": {"type": "integer", "minimum": 1000, "maximum": 500000},
                    "max_sample_chars": {"type": "integer", "minimum": 0, "maximum": 8000},
                    "sample_index": {"type": "integer", "minimum": 1, "maximum": 8},
                },
                "required": [
                    "hint",
                    "k",
                    "out_dir",
                    "validate",
                    "model",
                    "samples_per_role",
                    "max_new_tokens",
                    "min_new_tokens",
                    "temperature",
                    "top_p",
                    "repetition_penalty",
                    "stop_on_eot",
                    "max_items",
                    "max_chars",
                    "max_sample_chars",
                    "sample_index",
                ],
                "additionalProperties": False,
            },
            "strict": True,
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Investigator agent loop with tool access to context generation and target sampling."
    )
    parser.add_argument("--investigator-model", type=str, default="gpt-5")
    parser.add_argument("--aux-model", type=str, default="gpt-4")
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--max-output-tokens", type=int, default=16000)
    parser.add_argument("--parallel-tool-calls", action="store_true")
    parser.add_argument(
        "--sample-role",
        choices=["assistant", "user"],
        default="assistant",
        help="Which role/persona to sample from the target model and generate contexts for.",
    )
    parser.add_argument("--run-dir", type=Path, default=Path("agent_runs"))
    parser.add_argument(
        "--target-config",
        type=Path,
        default=Path("config.ini"),
        help="Path to config.ini with [target_model] model_id and optional adapter_id.",
    )
    args = parser.parse_args()

    _ensure_api_key()

    run_dir = _make_run_dir(args.run_dir)
    probe_runs_root = run_dir / "probe_runs"
    probe_runs_root.mkdir(parents=True, exist_ok=True)

    print(f"Run dir: {run_dir}")

    # Load target model once.
    qtmr.DEFAULT_CONFIG_PATH = args.target_config
    model_id, adapter_id, hf_cache_dir, allow_download = qtmr.load_target_model_config()
    hf_cache_dir = qtmr.configure_hf_cache(hf_cache_dir)
    device, dtype, backend = qtmr.detect_device()
    qtmr.device, qtmr.tok, qtmr.model = device, *qtmr.setup_model(
        device,
        dtype,
        model_id=model_id,
        adapter_id=adapter_id,
        hf_cache_dir=hf_cache_dir,
        allow_download=allow_download,
    )
    print(
        f"Target backend: {backend}, device {device}, dtype {dtype}, "
        f"model_id={model_id}, adapter_id={adapter_id or 'none'}, "
        f"hf_cache_dir={hf_cache_dir}, allow_download={allow_download}"
    )

    client = OpenAI()
    tools = build_tools_spec(args.sample_role)

    cfg = ToolConfig(
        aux_model=args.aux_model,
        target_output_root=run_dir,
        context_output_root=run_dir,
    )

    if args.sample_role == "assistant":
        investigator_prompt = INVESTIGATOR_AGENT_PROMPT_MULTIPLE_QUIRKS_ASSISTANT_SAMPLING_ONLY
    else:
        investigator_prompt = INVESTIGATOR_AGENT_PROMPT_MULTIPLE_QUIRKS_USER_SAMPLING_ONLY

    context: List[Dict[str, Any]] = [{"role": "user", "content": investigator_prompt}]
    response = client.responses.create(
        model=args.investigator_model,
        instructions=SYSTEM_PROMPT,
        input=context,
        tools=tools,
        tool_choice="auto",
        max_output_tokens=args.max_output_tokens,
        parallel_tool_calls=args.parallel_tool_calls,
    )

    history_path = run_dir / "conversation.jsonl"
    tool_calls_path = run_dir / "tool_calls.jsonl"
    with history_path.open("a", encoding="utf-8") as fh:
        fh.write(_safe_json_dumps(_response_to_loggable(response)) + "\n")

    for iteration in range(1, args.max_iterations + 1):
        tool_calls = list(_iter_function_calls(response))
        if not tool_calls:
            final_text = _extract_output_text(response)
            if final_text:
                print(final_text)
                (run_dir / "final_investigator_output.txt").write_text(final_text + "\n", encoding="utf-8")
            else:
                print("[no output text returned]")
                (run_dir / "final_investigator_output.txt").write_text(
                    "[no output text returned]\n",
                    encoding="utf-8",
                )
            return

        tool_outputs: List[Dict[str, Any]] = []
        for call in tool_calls:
            name = _get_attr(call, "name")
            call_id = _get_attr(call, "call_id")
            arguments = _get_attr(call, "arguments", "{}")
            raw_arguments = arguments
            if isinstance(arguments, str):
                try:
                    args_dict = json.loads(arguments)
                except json.JSONDecodeError:
                    args_dict = {}
            else:
                args_dict = dict(arguments)

            if name == "generate_and_query":
                probe_run_dir = _make_indexed_child_dir(probe_runs_root)
                call_cfg = ToolConfig(
                    aux_model=cfg.aux_model,
                    context_output_root=probe_run_dir / "generated_contexts",
                    target_output_root=probe_run_dir / "role_samples",
                )
                effective_generate_args = {
                    "hint": args_dict.get("hint", ""),
                    "k": int(args_dict.get("k", 1)),
                    "out_dir": "generated_contexts",
                    "validate": bool(args_dict.get("validate", False)),
                    "model": args_dict.get("model"),
                    "target_role": args.sample_role,
                }
                gen_result = generate_contexts_tool(
                    client,
                    call_cfg,
                    hint=effective_generate_args["hint"],
                    k=effective_generate_args["k"],
                    out_dir=effective_generate_args["out_dir"],
                    validate=effective_generate_args["validate"],
                    model=effective_generate_args["model"],
                    target_role=effective_generate_args["target_role"],
                )
                contexts_path = [gen_result.get("out_dir")] if gen_result.get("out_dir") else []
                effective_query_args = {
                    "contexts_path": contexts_path,
                    "role": args.sample_role,
                    "out_dir": "role_samples",
                    "samples_per_role": int(args_dict.get("samples_per_role", 1)),
                    "max_new_tokens": int(args_dict.get("max_new_tokens", 220)),
                    "min_new_tokens": int(args_dict.get("min_new_tokens", 100)),
                    "temperature": float(args_dict.get("temperature", 0.9)),
                    "top_p": float(args_dict.get("top_p", 0.95)),
                    "repetition_penalty": float(args_dict.get("repetition_penalty", 1.08)),
                    "stop_on_eot": bool(args_dict.get("stop_on_eot", True)),
                    "max_items": int(args_dict.get("max_items", 400)),
                    "max_chars": int(args_dict.get("max_chars", 180_000)),
                    "max_sample_chars": int(args_dict.get("max_sample_chars", 1500)),
                    "sample_index": int(args_dict.get("sample_index", 1)),
                }
                query_result = query_target_tool(
                    call_cfg,
                    contexts_path=effective_query_args["contexts_path"],
                    role=effective_query_args["role"],
                    out_dir=effective_query_args["out_dir"],
                    samples_per_role=effective_query_args["samples_per_role"],
                    max_new_tokens=effective_query_args["max_new_tokens"],
                    min_new_tokens=effective_query_args["min_new_tokens"],
                    temperature=effective_query_args["temperature"],
                    top_p=effective_query_args["top_p"],
                    repetition_penalty=effective_query_args["repetition_penalty"],
                    stop_on_eot=effective_query_args["stop_on_eot"],
                    max_items=effective_query_args["max_items"],
                    max_chars=effective_query_args["max_chars"],
                    max_sample_chars=effective_query_args["max_sample_chars"],
                    sample_index=effective_query_args["sample_index"],
                )
                with tool_calls_path.open("a", encoding="utf-8") as tcf:
                    tcf.write(
                        _safe_json_dumps(
                            {
                                "iteration": iteration,
                                "call_id": call_id,
                                "name": name,
                                "arguments_raw": raw_arguments,
                                "arguments_parsed": args_dict,
                                "effective_generate_args": effective_generate_args,
                                "effective_query_args": effective_query_args,
                            }
                        )
                        + "\n"
                    )
                result = {"generate": gen_result, "query": query_result}
            else:
                with tool_calls_path.open("a", encoding="utf-8") as tcf:
                    tcf.write(
                        _safe_json_dumps(
                            {
                                "iteration": iteration,
                                "call_id": call_id,
                                "name": name,
                                "arguments_raw": raw_arguments,
                                "arguments_parsed": args_dict,
                                "error": f"Unknown tool name: {name}",
                            }
                        )
                        + "\n"
                    )
                result = {"error": f"Unknown tool name: {name}"}

            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": _safe_json_dumps(result),
                }
            )

        context.extend(_output_items_to_input_items(response))
        context.extend(tool_outputs)
        response = client.responses.create(
            model=args.investigator_model,
            instructions=SYSTEM_PROMPT,
            input=context,
            tools=tools,
            tool_choice="auto",
            max_output_tokens=args.max_output_tokens,
            parallel_tool_calls=args.parallel_tool_calls,
        )

        with history_path.open("a", encoding="utf-8") as fh:
            fh.write(_safe_json_dumps(_response_to_loggable(response)) + "\n")

    # Force a final best-effort answer without allowing any more tool calls.
    # Do not replay unresolved function calls here; only carry assistant text if present.
    pending_text = _extract_output_text(response)
    if pending_text:
        context.append({"role": "assistant", "content": pending_text})
    context.append(
        {
            "role": "user",
            "content": (
                "Tool-call limit reached. Do not call tools. "
                "Provide your best final hypothesis now using the required format, "
                "based only on evidence collected so far."
            ),
        }
    )
    final_response = client.responses.create(
        model=args.investigator_model,
        instructions=SYSTEM_PROMPT,
        input=context,
        tools=tools,
        tool_choice="none",
        max_output_tokens=args.max_output_tokens,
    )
    with history_path.open("a", encoding="utf-8") as fh:
        fh.write(_safe_json_dumps(_response_to_loggable(final_response)) + "\n")

    final_text = _extract_output_text(final_response)
    if final_text:
        print(final_text)
        (run_dir / "final_investigator_output.txt").write_text(final_text + "\n", encoding="utf-8")
    else:
        print("[max iterations reached without final response]")
        (run_dir / "final_investigator_output.txt").write_text(
            "[max iterations reached without final response]\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
