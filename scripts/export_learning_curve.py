#!/usr/bin/env python3

"""Export learning-curve artifacts for a baseline run or a direct RL run."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _load_learning_curve_utils():
    repo_root = Path(__file__).resolve().parents[1]
    module_dir = repo_root / "source" / "isaaclab_eureka" / "isaaclab_eureka"
    sys.path.insert(0, str(module_dir))
    try:
        import learning_curve_utils  # type: ignore
    except ModuleNotFoundError as exc:
        missing_module = exc.name or "required dependency"
        raise SystemExit(
            f"Missing dependency '{missing_module}'. Run this script inside the project environment or via "
            "'uv run --with tensorboard --with matplotlib python scripts/export_learning_curve.py ...'."
        ) from exc

    return learning_curve_utils


def _resolve_run(input_path: str) -> tuple[str, str, str]:
    path = os.path.abspath(input_path)
    best_run_json = os.path.join(path, "best_run.json")
    if os.path.isfile(best_run_json):
        with open(best_run_json, "r") as metadata_file:
            metadata = json.load(metadata_file)
        training_log_dir = metadata.get("training_log_dir")
        training_run_dir = metadata.get("training_run_dir") or metadata.get("training_log_dir")
        if not training_log_dir or not training_run_dir:
            raise ValueError(f"Missing training_log_dir or training_run_dir in {best_run_json}")
        default_output_dir = os.path.join(path, "best_run_learning_curves")
        return os.path.abspath(training_log_dir), os.path.abspath(default_output_dir), "best_run"

    summaries_dir = os.path.join(path, "summaries")
    if os.path.isdir(summaries_dir):
        return os.path.abspath(summaries_dir), os.path.join(path, "learning_curves"), os.path.basename(path)

    return path, os.path.join(path, "learning_curves"), os.path.basename(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export learning curves for a baseline run or RL run directory.")
    parser.add_argument("input", help="Baseline run dir with best_run.json or direct RL run/log dir.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to the run's standard learning-curve folder.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional plot title override.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="default",
        choices=["default", "eureka", "oracle"],
        help="Metric to plot. Use 'oracle' for Eureka/oracle_total_rewards on the best RL run.",
    )
    args = parser.parse_args()

    learning_curve_utils = _load_learning_curve_utils()
    log_dir, default_output_dir, default_run_name = _resolve_run(args.input)
    output_dir = os.path.abspath(args.output_dir or default_output_dir)
    run_name = args.run_name or default_run_name

    result = learning_curve_utils.export_learning_curve_artifacts(
        log_dir=log_dir,
        output_dir=output_dir,
        run_name=run_name,
        metric=args.metric,
    )
    if result is None:
        raise RuntimeError(f"No scalar series matching metric '{args.metric}' were found in {log_dir}.")

    print(f"[INFO] Metric: {result['series_tag']}")
    print(f"[INFO] Saved plot: {result['plot_path']}")
    print(f"[INFO] Saved CSV: {result['csv_path']}")
    print(f"[INFO] Saved metadata: {result['metadata_path']}")


if __name__ == "__main__":
    main()
