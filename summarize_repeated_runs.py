"""Aggregate saved evaluation snapshots from run_repeated_experiments.py."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


METRICS = ("clip_comp", "clip_add", "blip_dino_official", "blip_atomic", "kid_set")
METRIC_LABELS = {
    "clip_comp": "CLIP-comp ↑",
    "clip_add": "CLIP-add ↑",
    "blip_dino_official": "BD(Off) ↑",
    "blip_atomic": "BLIP-At ↑",
    "kid_set": "Set-KID ↓",
}


def error_value(values: list[float], error: str) -> float:
    if len(values) < 2:
        return 0.0
    std = statistics.stdev(values)
    if error == "std":
        return std
    sem = std / math.sqrt(len(values))
    return sem if error == "sem" else 1.96 * sem


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize repeated evaluation runs as mean ± error.")
    parser.add_argument("experiment_root", type=Path, help="directory containing run_XXX/evaluation/evaluation_summary.json")
    parser.add_argument("--error", choices=("std", "sem", "ci95"), default="std")
    args = parser.parse_args()
    root = args.experiment_root.resolve()

    snapshots: list[tuple[str, dict]] = []
    for run_dir in sorted(path for path in root.glob("run_*") if path.is_dir()):
        snapshot = run_dir / "evaluation" / "evaluation_summary.json"
        if snapshot.exists():
            with snapshot.open(encoding="utf-8") as f:
                snapshots.append((run_dir.name, json.load(f)))

    if not snapshots:
        raise SystemExit(f"No completed evaluation snapshots found below {root}")

    methods: list[str] = []
    for _, snapshot in snapshots:
        for method in snapshot.get("methods", []):
            if method not in methods:
                methods.append(method)

    aggregate: dict[str, dict] = {}
    for method in methods:
        aggregate[method] = {}
        for metric in METRICS:
            values = [float(snapshot["final"][method][metric]) for _, snapshot in snapshots if method in snapshot.get("final", {})]
            aggregate[method][metric] = {
                "n": len(values),
                "values": values,
                "mean": statistics.mean(values) if values else 0.0,
                "error": error_value(values, args.error),
            }

    title = f"Repeated experiment summary (mean ± {args.error}; completed runs: {len(snapshots)})"
    headers = ["Method"] + [METRIC_LABELS[metric] for metric in METRICS]
    lines = [f"# {title}", "", " | ".join(headers), " | ".join(["---"] * len(headers))]
    for method in methods:
        row = [method]
        for metric in METRICS:
            data = aggregate[method][metric]
            row.append(f"{data['mean']:.5f} ± {data['error']:.5f} (n={data['n']})")
        lines.append(" | ".join(row))
    lines.extend(["", "`std` is the sample standard deviation; `sem` is standard error; `ci95` is 95% normal-approximation confidence half-width."])

    markdown_path = root / "repeated_summary.md"
    json_path = root / "repeated_summary.json"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps({"error_type": args.error, "completed_runs": [name for name, _ in snapshots], "summary": aggregate}, indent=2),
        encoding="utf-8",
    )
    print("\n".join(lines))
    print(f"\n[OK] Saved: {markdown_path}")
    print(f"[OK] Saved: {json_path}")


if __name__ == "__main__":
    main()
