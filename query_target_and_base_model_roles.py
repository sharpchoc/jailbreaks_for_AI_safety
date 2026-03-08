#!/usr/bin/env python3
"""Sample both adapter-enabled target and base-model continuations from saved contexts."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict

import query_target_model_roles as qtmr


def _base_sampling_context(model: Any):
    """Return a context manager that disables adapters for base-model sampling when available."""
    disable_adapter = getattr(model, "disable_adapter", None)
    if callable(disable_adapter):
        cm = disable_adapter()
        if hasattr(cm, "__enter__") and hasattr(cm, "__exit__"):
            return cm
    return nullcontext()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sample continuations from saved contexts using both the adapter-enabled target "
            "model and the same model with adapters disabled (base behavior)."
        )
    )
    parser.add_argument(
        "--contexts-path",
        type=Path,
        nargs="+",
        required=True,
        help="One or more directories/files with saved contexts.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("role_samples"), help="Base directory for outputs.")
    parser.add_argument("--target-config", type=Path, default=Path("config.ini"), help="Path to target config file.")
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
        f"Using backend {backend}, device {device}, dtype {dtype}, "
        f"model_id={model_id}, adapter_id={adapter_id or 'none'}, "
        f"hf_cache_dir={hf_cache_dir}, allow_download={allow_download}"
    )
    if adapter_id and not callable(getattr(qtmr.model, "disable_adapter", None)):
        raise RuntimeError(
            "Loaded model has an adapter but does not expose disable_adapter(). "
            "Upgrade peft or use a setup that supports temporary adapter disabling."
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
                    "model_config": {
                        "model_id": model_id,
                        "adapter_id": adapter_id,
                    },
                }

                target_samples = qtmr.sample_from_role_n(
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
                context_record["samples"][args.role] = {
                    "target_adapter": target_samples,
                }

                with _base_sampling_context(qtmr.model):
                    base_samples = qtmr.sample_from_role_n(
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
                context_record["samples"][args.role]["base_model"] = base_samples

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
