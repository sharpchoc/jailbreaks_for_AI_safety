import os
import argparse
import re
import json
from pathlib import Path
from typing import Literal

from openai import OpenAI
from prompts import CONTEXT_GEN_PROMPT_ASSISTANT_SAMPLING, CONTEXT_GEN_PROMPT_USER_SAMPLING
from utils.indexing import next_indexed_out_dir

SamplingMode = Literal["user", "assistant"]


CTX_BLOCK_RE = re.compile(r"<context>\s*(.*?)\s*</context>", re.DOTALL | re.IGNORECASE)

def extract_context_blocks(text: str) -> list[str]:
    return CTX_BLOCK_RE.findall(text)


def get_sampling_prompt_template(mode: SamplingMode) -> str:
    if mode == "assistant":
        return CONTEXT_GEN_PROMPT_ASSISTANT_SAMPLING
    elif mode == "user":
        return CONTEXT_GEN_PROMPT_USER_SAMPLING
    else:
        raise NameError(f"{mode} not recognized")


def extract_json_from_block(block: str) -> dict:
    """
    Tries to find the first JSON object in the block and parse it.
    Assumes each <context>...</context> contains exactly one JSON object.
    """
    # Find the first {...} blob (greedy across lines)
    m = re.search(r"\{.*\}", block, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found inside <context> block.")
    json_str = m.group(0)
    # special case where json str ends up having {{ ... }}
    if json_str.startswith("{{") and json_str.endswith("}}"):
        json_str = json_str[1:-1]
    return json.loads(json_str)

def save_contexts(context_dicts: list[dict], out_dir: Path, sampling_mode: str, raw_text: str | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save each as its own file
    for i, ctx in enumerate(context_dicts, start=1):
        p = out_dir / f"context_{sampling_mode}_{i:04d}.json"
        p.write_text(json.dumps(ctx, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Save the raw model output too (useful for debugging)
    if raw_text is not None:
        (out_dir / "raw_output.txt").write_text(raw_text, encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser()
    # parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--k", type=int, required=True)
    # parser.add_argument("--hint", type=str, default="generic")
    parser.add_argument("--hint", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-4")
    parser.add_argument(
        "--sampling-mode",
        choices=["user", "assistant"],
        default="assistant",
        help="Which context-generation prompt to use.",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY environment variable.")

    prompt_template = get_sampling_prompt_template(args.sampling_mode)
    prompt = prompt_template.replace("{{ HINT }}", args.hint).replace("{k}", str(args.k))

    client = OpenAI()
    resp = client.responses.create(model=args.model, input=prompt)
    text = resp.output_text

    blocks = extract_context_blocks(text)
    if len(blocks) != args.k:
        raise RuntimeError(f"Expected {args.k} <context> blocks, got {len(blocks)}.")

    context_dicts = []
    errors = []
    for idx, block in enumerate(blocks, start=1):
        try:
            context_dicts.append(extract_json_from_block(block))
        except Exception as e:
            errors.append((idx, str(e)))

    if errors:
        raise RuntimeError(f"Failed to parse JSON for contexts: {errors}")

    out_dir = next_indexed_out_dir(Path("generated_contexts"))
    save_contexts(context_dicts, out_dir, args.sampling_mode, raw_text=text)

    print(f"Saved {len(context_dicts)} contexts to: {out_dir} (sampling_mode={args.sampling_mode})")
    if errors:
        print(f"Warning: {len(errors)} contexts failed JSON parsing: {errors}")


if __name__ == "__main__":
    main()
