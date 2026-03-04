#!/usr/bin/env python3
"""Run investigator_agent_loop.py and then grade its final output."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run investigator_agent_loop.py and then grade_investigator_answers.py "
            "against the generated final output."
        )
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable to use.")
    parser.add_argument(
        "--investigator-script",
        type=Path,
        default=Path("investigator_agent_loop.py"),
        help="Path to investigator loop script.",
    )
    parser.add_argument(
        "--grader-script",
        type=Path,
        default=Path("grade_investigator_answers.py"),
        help="Path to grader script.",
    )
    parser.add_argument(
        "--answer-key-path",
        type=Path,
        default=Path("model_answers/rm_sycophancy_biases.txt"),
        help="Answer key passed to grader.",
    )
    parser.add_argument("--grader-model", default="gpt-5", help="Model for grader script.")
    parser.add_argument(
        "--grader-max-output-tokens",
        type=int,
        default=3000,
        help="Max output tokens for grader script.",
    )
    parser.add_argument(
        "investigator_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to investigator script. Use '--' before these args.",
    )
    return parser.parse_args()


def _extract_run_root(investigator_args: Sequence[str]) -> Path:
    run_root = Path("agent_runs")
    for idx, arg in enumerate(investigator_args):
        if arg == "--run-dir" and idx + 1 < len(investigator_args):
            return Path(investigator_args[idx + 1])
        if arg.startswith("--run-dir="):
            return Path(arg.split("=", 1)[1])
    return run_root


def _extract_num_runs(investigator_args: Sequence[str]) -> int:
    for idx, arg in enumerate(investigator_args):
        if arg == "--num-runs" and idx + 1 < len(investigator_args):
            try:
                return max(1, int(investigator_args[idx + 1]))
            except ValueError:
                return 1
        if arg.startswith("--num-runs="):
            try:
                return max(1, int(arg.split("=", 1)[1]))
            except ValueError:
                return 1
    return 1


def _discover_latest_output(run_root: Path) -> Path | None:
    if not run_root.exists():
        return None
    candidates = [
        p for p in run_root.rglob("final_investigator_output.txt") if p.is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_investigator(
    python_exec: str,
    script_path: Path,
    investigator_args: Sequence[str],
) -> list[Path]:
    run_dirs: list[Path] = []
    cmd = [python_exec, str(script_path), *investigator_args]

    print(f"[runner] Executing investigator: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        if line.startswith("Run dir:"):
            run_dirs.append(Path(line.split("Run dir:", 1)[1].strip()))

    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Investigator script failed with exit code {rc}.")

    if not run_dirs:
        run_root = _extract_run_root(investigator_args)
        latest = _discover_latest_output(run_root)
        if latest is None:
            raise RuntimeError(
                "Could not determine run directory from investigator output and could not find "
                "any final_investigator_output.txt under run root."
            )
        return [latest]

    deduped_run_dirs: list[Path] = []
    seen = set()
    for rd in run_dirs:
        key = str(rd)
        if key in seen:
            continue
        seen.add(key)
        deduped_run_dirs.append(rd)

    expected_runs = _extract_num_runs(investigator_args)
    if expected_runs > 1 and len(deduped_run_dirs) != expected_runs:
        raise RuntimeError(
            f"Expected {expected_runs} run dirs from investigator output, got {len(deduped_run_dirs)}."
        )

    final_outputs: list[Path] = []
    for run_dir in deduped_run_dirs:
        final_output = run_dir / "final_investigator_output.txt"
        if not final_output.exists():
            raise RuntimeError(f"Expected output not found: {final_output}")
        final_outputs.append(final_output)
    return final_outputs


def run_grader(
    python_exec: str,
    script_path: Path,
    investigator_output_path: Path,
    answer_key_path: Path,
    model: str,
    max_output_tokens: int,
) -> None:
    cmd = [
        python_exec,
        str(script_path),
        "--investigator-output-path",
        str(investigator_output_path),
        "--answer-key-path",
        str(answer_key_path),
        "--model",
        model,
        "--max-output-tokens",
        str(max_output_tokens),
    ]

    print(f"[runner] Executing grader: {' '.join(cmd)}")
    rc = subprocess.run(cmd, check=False).returncode
    if rc != 0:
        raise RuntimeError(f"Grader script failed with exit code {rc}.")


def main() -> None:
    args = parse_args()

    passthrough = list(args.investigator_args)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    final_output_paths = run_investigator(
        python_exec=args.python,
        script_path=args.investigator_script,
        investigator_args=passthrough,
    )

    for final_output_path in final_output_paths:
        run_grader(
            python_exec=args.python,
            script_path=args.grader_script,
            investigator_output_path=final_output_path,
            answer_key_path=args.answer_key_path,
            model=args.grader_model,
            max_output_tokens=args.grader_max_output_tokens,
        )
        grading_path = final_output_path.parent / "grading.json"
        print(f"[runner] Investigator output: {final_output_path}")
        print(f"[runner] Grading JSON: {grading_path}")


if __name__ == "__main__":
    main()
