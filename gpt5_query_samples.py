#!/usr/bin/env python3
"""Query sampled model responses with GPT-5 using the OpenAI API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from openai import OpenAI
from prompts import AUDITING_SYSTEM_PROMPT, RESPONSE_GRADER_PROMPT

SYSTEM_PROMPT = AUDITING_SYSTEM_PROMPT
USER_PROMPT = RESPONSE_GRADER_PROMPT


def _iter_records(path: Path) -> Iterable[Dict[str, Any]]:
    if path.is_dir():
        jsonl = path / "responses.jsonl"
        if jsonl.exists():
            yield from _iter_records(jsonl)
            return
        context_files = sorted(path.glob("context_*.json"))
        if context_files:
            for file in context_files:
                yield from _iter_records(file)
            return
        subdirs = [p for p in sorted(path.iterdir()) if p.is_dir()]
        if subdirs:
            for subdir in subdirs:
                yield from _iter_records(subdir)
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


def collect_user_samples(
    paths: Iterable[Path],
    user_sample_index: int,
) -> List[Tuple[str, str, int, str, str]]:
    samples: List[Tuple[str, str, int, str, str]] = []
    for path in paths:
        for record in _iter_records(path):
            context_id = str(record.get("context_id", "unknown_context"))
            messages = record.get("messages", []) or []
            per_role = record.get("samples", {}) or {}
            user_items = per_role.get("user", []) or []
            if not user_items:
                continue
            idx = max(1, user_sample_index)
            if idx > len(user_items):
                continue
            item = user_items[idx - 1]
            text = str(item.get("generated_text", "")).strip()
            if text:
                samples.append((context_id, "user", idx, text, json.dumps(messages, ensure_ascii=False)))
    return samples


def build_corpus(
    samples: List[Tuple[str, str, int, str, str]],
    *,
    max_items: int,
    max_chars: int,
    max_sample_chars: int | None,
) -> str:
    lines: List[str] = []
    total_chars = 0
    for i, (context_id, role, sample_idx, text, messages_json) in enumerate(samples[:max_items], start=1):
        if max_sample_chars is not None and len(text) > max_sample_chars:
            text = text[:max_sample_chars].rstrip()
        header = (
            f"[{i}] source=target_model context={context_id} role={role} "
            f"resample_index={sample_idx}"
        )
        block = (
            f"{header}\n"
            "TARGET_MODEL_INPUT_BEGIN\n"
            f"{messages_json}\n"
            "TARGET_MODEL_INPUT_END\n"
            "TARGET_MODEL_OUTPUT_BEGIN\n"
            f"{text}\n"
            "TARGET_MODEL_OUTPUT_END\n"
        )
        if total_chars + len(block) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 0:
                lines.append(block[:remaining])
            break
        lines.append(block)
        total_chars += len(block)
    return "\n".join(lines).strip()


def extract_output_text(response: Any) -> Tuple[str, str]:
    text = getattr(response, "output_text", "") or ""
    if text:
        return text, "output_text"
    outputs = getattr(response, "output", []) or []
    for item in outputs:
        contents = getattr(item, "content", []) or []
        for content in contents:
            if getattr(content, "type", "") == "output_text":
                candidate = getattr(content, "text", "") or ""
                if candidate:
                    return candidate, "output.content"
    return "", "missing"


def main() -> None:
    parser = argparse.ArgumentParser(description="Query sampled responses with GPT-5.")
    parser.add_argument(
        "--samples-path",
        type=Path,
        nargs="+",
        required=True,
        help="One or more paths to responses.jsonl or directories.",
    )
    parser.add_argument("--system-prompt", type=str, default=None, help="System prompt for GPT-5.")
    parser.add_argument("--user-prompt", type=str, default=None, help="User prompt/query to answer.")
    parser.add_argument("--model", type=str, default="gpt-5", help="Model id (default: gpt-5).")
    parser.add_argument("--max-items", type=int, default=400, help="Maximum samples to include.")
    parser.add_argument("--max-chars", type=int, default=180_000, help="Max characters of samples in context.")
    parser.add_argument(
        "--max-sample-chars",
        type=int,
        default=1500,
        help="Max characters per sample (set to 0 to disable).",
    )
    parser.add_argument(
        "--user-sample-index",
        type=int,
        default=1,
        help="Which user sample to use per context (1-based).",
    )
    parser.add_argument("--max-output-tokens", type=int, default=1200, help="Max output tokens.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to write a JSON record of the model output.",
    )
    args = parser.parse_args()

    samples = collect_user_samples(args.samples_path, args.user_sample_index)
    if not samples:
        raise RuntimeError("No samples found in the provided path.")

    max_sample_chars = None if args.max_sample_chars == 0 else args.max_sample_chars
    corpus = build_corpus(
        samples,
        max_items=args.max_items,
        max_chars=args.max_chars,
        max_sample_chars=max_sample_chars,
    )
    system_prompt = args.system_prompt if args.system_prompt is not None else SYSTEM_PROMPT
    user_prompt = args.user_prompt if args.user_prompt is not None else USER_PROMPT

    user_text = (
        f"{user_prompt}\n\n"
        "SAMPLED RESPONSES:\n"
        f"{corpus}\n\n"
        "Answer using only the sampled responses above. "
        "If the samples do not contain enough information, say so."
    )

    client = OpenAI()
    response = client.responses.create(
        model=args.model,
        instructions=system_prompt,
        input=user_text,
        max_output_tokens=args.max_output_tokens,
    )

    output_text, output_source = extract_output_text(response)
    incomplete_details = getattr(response, "incomplete_details", None)
    if output_text:
        print(output_text)
    else:
        print("[no output text returned]")
    print(f"[output_text_source={output_source}]")
    response_id = getattr(response, "id", None)
    response_status = getattr(response, "status", None)
    if response_id:
        print(f"[response_id={response_id}]")
    if response_status:
        print(f"[response_status={response_status}]")
    if incomplete_details:
        reason = getattr(incomplete_details, "reason", None)
        if reason:
            print(f"[incomplete_reason={reason}]")

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "model": args.model,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "samples_path": str(args.samples_path),
            "max_items": args.max_items,
            "max_chars": args.max_chars,
            "max_sample_chars": args.max_sample_chars,
            "user_sample_index": args.user_sample_index,
            "max_output_tokens": args.max_output_tokens,
            "num_samples_used": min(len(samples), args.max_items),
            "sample_ids": [
                {
                    "context_id": context_id,
                    "role": role,
                    "sample_index": sample_idx,
                }
                for context_id, role, sample_idx, _, _ in samples[: args.max_items]
            ],
            "output_text": output_text,
            "output_text_source": output_source,
            "response_id": response_id,
            "response_status": response_status,
            "incomplete_reason": getattr(incomplete_details, "reason", None) if incomplete_details else None,
        }
        output_path = args.output_dir / "hypothesis.json"
        output_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
