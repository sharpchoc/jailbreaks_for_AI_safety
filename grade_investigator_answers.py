#!/usr/bin/env python3
"""Grade investigator hypotheses against an answer-key text file using GPT-5."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Tuple

from openai import OpenAI

from prompts import (
    HYPOTHESIS_GRADING_SYSTEM_PROMPT,
    HYPOTHESIS_GRADING_USER_PROMPT_TEMPLATE,
)


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


def response_debug_info(response: Any) -> Dict[str, Any]:
    incomplete_details = getattr(response, "incomplete_details", None)
    return {
        "response_id": getattr(response, "id", None),
        "status": getattr(response, "status", None),
        "incomplete_reason": getattr(incomplete_details, "reason", None) if incomplete_details else None,
        "output_items": len(getattr(response, "output", []) or []),
    }


def extract_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fence_match:
        parsed = json.loads(fence_match.group(1))
        if isinstance(parsed, dict):
            return parsed

    obj_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if obj_match:
        parsed = json.loads(obj_match.group(0))
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("Could not parse a JSON object from model output.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grade investigator output text against answer key via GPT-5."
    )
    parser.add_argument(
        "--investigator-output-path",
        type=Path,
        required=True,
        help="Path to investigator output .txt file (e.g. final_investigator_output.txt).",
    )
    parser.add_argument(
        "--answer-key-path",
        type=Path,
        default=Path("model_answers/rm_sycophancy_biases.txt"),
        help="Path to answer key text file.",
    )
    parser.add_argument("--model", type=str, default="gpt-5", help="Model id (default: gpt-5).")
    parser.add_argument("--max-output-tokens", type=int, default=2000, help="Max output tokens.")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY environment variable.")
    if not args.investigator_output_path.exists():
        raise FileNotFoundError(f"Investigator output file not found: {args.investigator_output_path}")
    if not args.answer_key_path.exists():
        raise FileNotFoundError(f"Answer key file not found: {args.answer_key_path}")

    investigator_output_text = args.investigator_output_path.read_text(encoding="utf-8")
    answer_key_text = args.answer_key_path.read_text(encoding="utf-8")
    user_prompt = HYPOTHESIS_GRADING_USER_PROMPT_TEMPLATE.format(
        answer_key_text=answer_key_text,
        investigator_output_text=investigator_output_text,
    )

    client = OpenAI()
    response = client.responses.create(
        model=args.model,
        instructions=HYPOTHESIS_GRADING_SYSTEM_PROMPT,
        input=user_prompt,
        max_output_tokens=args.max_output_tokens,
    )

    output_text, output_source = extract_output_text(response)
    if not output_text:
        # Retry once with a larger budget and tighter instruction when model returns no text.
        retry_prompt = user_prompt + "\n\nReturn only the final JSON object."
        retry_response = client.responses.create(
            model=args.model,
            instructions=HYPOTHESIS_GRADING_SYSTEM_PROMPT,
            input=retry_prompt,
            max_output_tokens=max(3000, args.max_output_tokens * 2),
        )
        output_text, output_source = extract_output_text(retry_response)
        if not output_text:
            first_debug = response_debug_info(response)
            second_debug = response_debug_info(retry_response)
            raise RuntimeError(
                "No output text returned from model after retry. "
                f"first_response={first_debug}, retry_response={second_debug}"
            )
        response = retry_response
    grading = extract_json_object(output_text)
    grading.pop("missed_answer_ids", None)

    print(json.dumps(grading, ensure_ascii=False, indent=2))
    print(f"[output_text_source={output_source}]")
    response_id = getattr(response, "id", None)
    if response_id:
        print(f"[response_id={response_id}]")

    output_json = args.investigator_output_path.parent / "grading.json"
    output_json.write_text(
        json.dumps(
            {
                "model": args.model,
                "investigator_output_path": str(args.investigator_output_path),
                "answer_key_path": str(args.answer_key_path),
                "grading": grading,
                "raw_output_text": output_text,
                "output_text_source": output_source,
                "response_id": response_id,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[saved_json={output_json}]")


if __name__ == "__main__":
    main()
