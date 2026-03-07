#!/usr/bin/env python3
"""Sample model continuations from saved contexts using the base model only."""

from __future__ import annotations

import argparse
from pathlib import Path

import query_target_model_roles as qtmr


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample both user and assistant continuations from saved contexts (base model only)."
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

    qtmr.DEFAULT_CONFIG_PATH = qtmr.DEFAULT_CONFIG_PATH

    model_id, _adapter_id, hf_cache_dir, allow_download = qtmr.load_target_model_config()
    hf_cache_dir = qtmr.configure_hf_cache(hf_cache_dir)
    device, dtype, backend = qtmr.detect_device()

    qtmr.device, qtmr.tok, qtmr.model = device, *qtmr.setup_model(
        device,
        dtype,
        model_id=model_id,
        adapter_id=None,
        hf_cache_dir=hf_cache_dir,
        allow_download=allow_download,
    )

    print(
        f"Using backend {backend}, device {device}, dtype {dtype}, "
        f"model_id={model_id}, adapter_id=none, "
        f"hf_cache_dir={hf_cache_dir}, allow_download={allow_download}"
    )

    contexts = qtmr.collect_contexts_from_paths(args.contexts_path)
    if not contexts:
        raise RuntimeError("No contexts found to sample from.")
    qtmr._validate_context_role_match(contexts, args.role)

    output_roots_by_label = {}
    contexts_by_label = {}
    for label, context_id, context in contexts:
        contexts_by_label.setdefault(label, []).append((context_id, context))

    total_contexts = 0
    for label, labeled_contexts in contexts_by_label.items():
        total_contexts += len(labeled_contexts)
        labeled_output_root = qtmr.next_indexed_out_dir(args.output_dir)
        labeled_output_root.mkdir(parents=True, exist_ok=True)
        output_roots_by_label[label] = labeled_output_root
        aggregated_path = labeled_output_root / "responses.jsonl"

        with aggregated_path.open("w", encoding="utf-8") as aggregate_handle:
            for context_id, context in labeled_contexts:
                messages = qtmr.build_messages_from_context(context)

                context_record = {
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

                role_samples = qtmr.sample_from_role_n(
                    messages,
                    args.role,
                    num_samples=args.samples_per_role,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=args.min_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    stop_on_eot=not args.no_stop_on_eot,
                )
                context_record["samples"][args.role] = role_samples

                per_context_path = labeled_output_root / f"generated_{args.role}_{context_id[-4:]}.json"
                per_context_path.write_text(
                    qtmr.json.dumps(context_record, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                aggregate_handle.write(qtmr.json.dumps(context_record, ensure_ascii=False) + "\n")

    roots_summary = ", ".join(
        f"{label} -> {path}" for label, path in sorted(output_roots_by_label.items())
    )
    print(f"Sampled {total_contexts} contexts ↦ {roots_summary}")


if __name__ == "__main__":
    main()
