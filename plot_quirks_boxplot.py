#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class RunScore:
    run_id: str
    correct_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a box plot of identified quirks (grading.correct_count) by category."
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("experiment_runs"),
        help="Root folder containing run subfolders with grading.json files.",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("experiment_runs/category_mapping.json"),
        help="JSON file mapping categories to run IDs (e.g., 20260305/0001).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiment_runs/plots/quirks_boxplot.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--title",
        default="Number of Identified Quirks by Category",
        help="Plot title.",
    )
    parser.add_argument(
        "--max-quirks",
        type=int,
        default=51,
        help="Maximum number of quirks in the answer key (for y-axis label).",
    )
    return parser.parse_args()


def extract_correct_count(grading_path: Path) -> int:
    data = json.loads(grading_path.read_text())
    if isinstance(data, dict) and "grading" in data and isinstance(data["grading"], dict):
        value = data["grading"].get("correct_count")
    else:
        value = data.get("correct_count") if isinstance(data, dict) else None

    if value is None:
        raise ValueError(f"No correct_count found in {grading_path}")
    return int(value)


def load_scores(runs_root: Path) -> Dict[str, RunScore]:
    scores: Dict[str, RunScore] = {}
    for grading_path in sorted(runs_root.glob("*/*/grading.json")):
        run_dir = grading_path.parent
        run_id = run_dir.relative_to(runs_root).as_posix()
        scores[run_id] = RunScore(run_id=run_id, correct_count=extract_correct_count(grading_path))
    return scores


def normalize_mapping(raw: object) -> List[Tuple[str, List[str]]]:
    if isinstance(raw, dict) and "categories" in raw:
        categories = raw["categories"]
        out: List[Tuple[str, List[str]]] = []
        if not isinstance(categories, list):
            raise ValueError("mapping['categories'] must be a list")
        for entry in categories:
            if not isinstance(entry, dict):
                raise ValueError("Each category entry must be an object")
            name = entry.get("name")
            runs = entry.get("runs")
            if not isinstance(name, str) or not isinstance(runs, list):
                raise ValueError("Each category entry must have string 'name' and list 'runs'")
            out.append((name, [str(r) for r in runs]))
        return out

    if isinstance(raw, dict):
        out = []
        for name, runs in raw.items():
            if not isinstance(name, str) or not isinstance(runs, list):
                raise ValueError("Mapping must be {category_name: [run_ids...]} or use 'categories'.")
            out.append((name, [str(r) for r in runs]))
        return out

    raise ValueError("Unsupported mapping format.")


def load_category_mapping(mapping_path: Path) -> List[Tuple[str, List[str]]]:
    raw = json.loads(mapping_path.read_text())
    return normalize_mapping(raw)


def build_plot_data(
    categories: Iterable[Tuple[str, List[str]]], scores: Dict[str, RunScore]
) -> Tuple[List[str], List[List[int]], List[str]]:
    category_names: List[str] = []
    values: List[List[int]] = []
    missing: List[str] = []

    for name, run_ids in categories:
        collected: List[int] = []
        for run_id in run_ids:
            if run_id in scores:
                collected.append(scores[run_id].correct_count)
            else:
                missing.append(run_id)
        if collected:
            category_names.append(name)
            values.append(collected)

    return category_names, values, missing


def make_plot(
    category_names: List[str],
    values: List[List[int]],
    output_path: Path,
    title: str,
    max_quirks: int,
) -> Path:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError:
        actual_output = output_path.with_suffix(".svg")
        _make_svg_plot(category_names, values, actual_output, title, max_quirks)
        return actual_output

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    fig.patch.set_facecolor("#f3f3f3")
    ax.set_facecolor("#f7f7f7")

    positions = np.arange(1, len(values) + 1)
    bp = ax.boxplot(
        values,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        medianprops={"color": "#1f1f1f", "linewidth": 1.8},
        boxprops={"linewidth": 1.2, "edgecolor": "#222"},
        whiskerprops={"linewidth": 1.2, "color": "#222"},
        capprops={"linewidth": 1.2, "color": "#222"},
        flierprops={
            "marker": "o",
            "markerfacecolor": "none",
            "markeredgecolor": "#333",
            "markersize": 4.5,
            "alpha": 0.9,
        },
    )

    palette = plt.cm.Set2(np.linspace(0.1, 0.9, len(values)))
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)

    rng = np.random.default_rng(7)
    for x, ys in zip(positions, values):
        jitter_x = rng.normal(loc=x, scale=0.045, size=len(ys))
        ax.scatter(jitter_x, ys, s=18, alpha=0.55, color="#2f2f2f", linewidths=0)

    y_min = min(min(v) for v in values)
    y_max = max(max(v) for v in values)
    y_span = max(y_max - y_min, 1)
    label_y = y_min - 0.08 * y_span

    for x, ys in zip(positions, values):
        ax.text(x, label_y, f"N={len(ys)}", ha="center", va="top", fontsize=9, color="#333")

    ax.set_xticks(positions)
    ax.set_xticklabels(category_names, rotation=18, ha="right")
    ax.set_title(title, fontsize=16, color="#b65a3a")
    ax.set_ylabel(f"Number of Quirks Found (out of {max_quirks})", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    ax.set_ylim(bottom=min(0, label_y - 0.2 * y_span), top=max(max_quirks, y_max + 1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _make_svg_plot(
    category_names: List[str],
    values: List[List[int]],
    output_path: Path,
    title: str,
    max_quirks: int,
) -> None:
    width, height = 1400, 800
    margin_left, margin_right = 120, 40
    margin_top, margin_bottom = 90, 180
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    y_min = 0.0
    y_max = float(max(max_quirks, max(max(v) for v in values) + 1))
    x_count = len(values)

    def y_to_px(y: float) -> float:
        return margin_top + (y_max - y) * (plot_h / (y_max - y_min))

    def x_center(i: int) -> float:
        return margin_left + (i + 0.5) * (plot_w / x_count)

    colors = [
        "#8dd3c7",
        "#ffffb3",
        "#bebada",
        "#fb8072",
        "#80b1d3",
        "#fdb462",
        "#b3de69",
        "#fccde5",
    ]
    box_w = min(90.0, (plot_w / x_count) * 0.45)
    rng = np.random.default_rng(7)
    lines: List[str] = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="#f3f3f3"/>')
    lines.append(
        f'<text x="{width/2:.1f}" y="48" text-anchor="middle" font-family="sans-serif" font-size="38" fill="#b65a3a">{title}</text>'
    )
    lines.append(
        f'<text x="34" y="{height/2:.1f}" transform="rotate(-90 34,{height/2:.1f})" text-anchor="middle" font-family="sans-serif" font-size="30" fill="#222">Number of Quirks Found (out of {max_quirks})</text>'
    )
    for tick in range(int(y_min), int(y_max) + 1, 5):
        y = y_to_px(float(tick))
        lines.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width-margin_right}" y2="{y:.2f}" stroke="#c9c9c9" stroke-dasharray="6,6" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{margin_left-12}" y="{y+5:.2f}" text-anchor="end" font-family="sans-serif" font-size="20" fill="#444">{tick}</text>'
        )
    lines.append(
        f'<line x1="{margin_left}" y1="{y_to_px(0):.2f}" x2="{width-margin_right}" y2="{y_to_px(0):.2f}" stroke="#444" stroke-width="2"/>'
    )
    for i, ys in enumerate(values):
        c = x_center(i)
        data = np.asarray(ys, dtype=float)
        q1, med, q3 = np.percentile(data, [25, 50, 75])
        iqr = q3 - q1
        low_thr = q1 - 1.5 * iqr
        high_thr = q3 + 1.5 * iqr
        inliers = data[(data >= low_thr) & (data <= high_thr)]
        low_whisk = float(np.min(inliers)) if len(inliers) else float(np.min(data))
        high_whisk = float(np.max(inliers)) if len(inliers) else float(np.max(data))
        outliers = data[(data < low_thr) | (data > high_thr)]
        yq1, ymed, yq3 = y_to_px(q1), y_to_px(med), y_to_px(q3)
        ylow, yhigh = y_to_px(low_whisk), y_to_px(high_whisk)
        color = colors[i % len(colors)]
        lines.append(
            f'<rect x="{c-box_w/2:.2f}" y="{yq3:.2f}" width="{box_w:.2f}" height="{(yq1-yq3):.2f}" fill="{color}" fill-opacity="0.88" stroke="#222" stroke-width="2"/>'
        )
        lines.append(f'<line x1="{c-box_w/2:.2f}" y1="{ymed:.2f}" x2="{c+box_w/2:.2f}" y2="{ymed:.2f}" stroke="#111" stroke-width="3"/>')
        lines.append(f'<line x1="{c:.2f}" y1="{yq3:.2f}" x2="{c:.2f}" y2="{yhigh:.2f}" stroke="#222" stroke-width="2"/>')
        lines.append(f'<line x1="{c:.2f}" y1="{yq1:.2f}" x2="{c:.2f}" y2="{ylow:.2f}" stroke="#222" stroke-width="2"/>')
        lines.append(f'<line x1="{c-24:.2f}" y1="{yhigh:.2f}" x2="{c+24:.2f}" y2="{yhigh:.2f}" stroke="#222" stroke-width="2"/>')
        lines.append(f'<line x1="{c-24:.2f}" y1="{ylow:.2f}" x2="{c+24:.2f}" y2="{ylow:.2f}" stroke="#222" stroke-width="2"/>')
        for y in ys:
            xj = c + float(rng.normal(0, box_w * 0.11))
            yj = y_to_px(float(y))
            lines.append(f'<circle cx="{xj:.2f}" cy="{yj:.2f}" r="3.2" fill="#2f2f2f" fill-opacity="0.55"/>')
        for y in outliers:
            yo = y_to_px(float(y))
            lines.append(f'<circle cx="{c:.2f}" cy="{yo:.2f}" r="4.3" fill="none" stroke="#333" stroke-width="1.6"/>')
        lines.append(
            f'<text x="{c:.2f}" y="{height-margin_bottom+40}" text-anchor="middle" font-family="sans-serif" font-size="22" fill="#333" transform="rotate(18 {c:.2f},{height-margin_bottom+40})">{category_names[i]}</text>'
        )
        lines.append(
            f'<text x="{c:.2f}" y="{height-margin_bottom+80}" text-anchor="middle" font-family="sans-serif" font-size="20" fill="#333">N={len(ys)}</text>'
        )
    lines.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    scores = load_scores(args.runs_root)
    if not scores:
        raise SystemExit(f"No grading.json files found under {args.runs_root}")

    categories = load_category_mapping(args.mapping)
    category_names, values, missing = build_plot_data(categories, scores)

    if not values:
        raise SystemExit("No mapped runs were found. Check the run IDs in the mapping file.")

    actual_output = make_plot(category_names, values, args.output, args.title, args.max_quirks)

    if missing:
        print("Warning: some mapped runs were not found:")
        for run_id in missing:
            print(f"  - {run_id}")

    print(f"Saved plot to: {actual_output}")
    for name, vals in zip(category_names, values):
        print(f"{name}: N={len(vals)}, median={np.median(vals):.2f}, mean={np.mean(vals):.2f}")


if __name__ == "__main__":
    main()
