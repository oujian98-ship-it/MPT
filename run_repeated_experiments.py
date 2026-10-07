"""Run reproducible repeated experiments and retain every run's artifacts.

Examples:
  python -u run_repeated_experiments.py
  python -u run_repeated_experiments.py --mode all --repeats 5
  python -u run_repeated_experiments.py --mode mpt --repeats 3 --seed 20260907
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def choose_interactively(args: argparse.Namespace) -> None:
    if args.mode is None:
        print("\nSelect the experiment scope:")
        print("  1. All Methods (all baselines + MPT)")
        print("  2. Only MPT (report MPT only; generate vanilla references for evaluation)")
        while True:
            selection = input("Enter 1 or 2 [1]: ").strip() or "1"
            if selection == "1":
                args.mode = "all"
                break
            if selection == "2":
                args.mode = "mpt"
                break
            print("Please enter 1 or 2.")

    if args.repeats is None:
        while True:
            raw = input("Repetitions (recommended: 3 or 5)[3]: ").strip() or "3"
            try:
                args.repeats = positive_int(raw)
                break
            except (ValueError, argparse.ArgumentTypeError):
                print("Please enter a positive integer, such as 3 or 5.")


def run_command(command: list[str], env: dict[str, str]) -> None:
    print("\n>>> " + " ".join(command))
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeat All Methods or MPT experiments and aggregate mean ± error.")
    parser.add_argument("--mode", choices=("all", "mpt"), help="all = baselines + MPT; mpt = MPT only")
    parser.add_argument("--repeats", type=positive_int, help="number of independent runs (for example 3 or 5)")
    parser.add_argument("--seed", type=int, default=20260907, help="base seed; each repetition receives a distinct seed")
    parser.add_argument("--output-root", type=Path, help="directory in which all run artifacts are kept")
    parser.add_argument("--error", choices=("std", "sem", "ci95"), default="sem", help="error reported in the final table (default: sem)")
    args = parser.parse_args()
    choose_interactively(args)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (args.output_root or Path("experiment_results") / f"{args.mode}_{args.repeats}runs_{timestamp}").resolve()
    output_root.mkdir(parents=True, exist_ok=False)

    manifest = {
        "mode": args.mode,
        "repeats": args.repeats,
        "base_seed": args.seed,
        "error": args.error,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "note": "Each run is retained in run_XXX. Mean/error use final metrics from evaluation_summary.json.",
    }
    (output_root / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nResults will be retained in:{output_root}")
    for run_number in range(1, args.repeats + 1):
        run_dir = output_root / f"run_{run_number:03d}"
        images_dir = run_dir / "images"
        evaluation_dir = run_dir / "evaluation"
        log_dir = run_dir / "logs"
        for directory in (images_dir, evaluation_dir, log_dir):
            directory.mkdir(parents=True, exist_ok=True)

        run_seed = args.seed + (run_number - 1) * 10_000_000
        run_meta = {"run": run_number, "seed": run_seed, "mode": args.mode}
        (run_dir / "run_config.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
        env = os.environ.copy()
        env.update({
            "RESULTS_DIR": str(images_dir),
            "EVAL_OUTPUT_DIR": str(evaluation_dir),
            "LOG_DIR": str(log_dir),
            "EXPERIMENT_SEED": str(run_seed),
            # MPT-only still needs vanilla reference images (text1~4) for DINO/KID.
            "METHODS_TO_RUN": "all" if args.mode == "all" else "vanilla",
            "METHODS": "all" if args.mode == "all" else "mpt",
        })

        print(f"\n{'=' * 78}\nStarting run {run_number}/{args.repeats}, seed={run_seed}\n{'=' * 78}")
        try:
            run_command([sys.executable, "-u", "run_all_unified.py"], env)
            run_command([sys.executable, "-u", "run_batch_mpt.py"], env)
            run_command([sys.executable, "-u", "eval_per_set.py"], env)
        except subprocess.CalledProcessError as exc:
            print(f"\n[FAILED] Run {run_number} failed (exit code {exc.returncode}). Generated files are retained in:{run_dir}")
            raise SystemExit(exc.returncode) from exc

    run_command(
        [sys.executable, "-u", "summarize_repeated_runs.py", str(output_root), "--error", args.error],
        os.environ.copy(),
    )
    print(f"\n[DONE] All runs complete. Final summary:{output_root / 'repeated_summary.md'}")


if __name__ == "__main__":
    main()
