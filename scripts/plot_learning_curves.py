#!/usr/bin/env python3

"""Overlay learning curves from baseline run directories or raw RL run directories."""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path
from typing import Any

DEFAULT_BASELINES = ("eureka", "revolve", "revolve_full", "tacreka_sr")
BASELINE_ALIASES = {
    "eureak": "eureka",
    "revolve-full": "revolve_full",
    "revolvefull": "revolve_full",
    "tacreka": "tacreka_sr",
    "tacreaka": "tacreka_sr",
}


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
            "'uv run --with tensorboard --with matplotlib python scripts/plot_learning_curves.py ...'."
        ) from exc

    return learning_curve_utils


def _normalize_baseline_name(name: str) -> str:
    normalized_name = name.strip().lower()
    return BASELINE_ALIASES.get(normalized_name, normalized_name)


def _label_for_path(path: str) -> str:
    resolved_path = Path(path).resolve()
    path_parts = resolved_path.parts
    if "logs" in path_parts:
        logs_index = path_parts.index("logs")
        relative_parts = path_parts[logs_index + 1 :]
        if len(relative_parts) >= 3:
            family, task, run_name = relative_parts[:3]
            return f"{family}/{task}/{run_name}"
    return resolved_path.name


def _slugify_task_name(task_name: str) -> str:
    normalized_name = task_name.strip()
    if normalized_name == "Isaac-Cartpole-Direct-v0":
        return "cartpole"
    if normalized_name == "Isaac-Quadcopter-Direct-v0":
        return "quadcopter"
    return normalized_name.replace(os.sep, "_").replace(" ", "_")


def _normalize_log_dirs(value: Any, context: str) -> str | list[str]:
    if isinstance(value, str):
        return os.path.abspath(value)
    if isinstance(value, (list, tuple)):
        normalized_paths = [os.path.abspath(str(path)) for path in value if path]
        if normalized_paths:
            return normalized_paths
    raise ValueError(f"Could not resolve log directory information from {context}")


def _resolve_log_dir(input_path: str) -> tuple[str | list[str], str]:
    path = os.path.abspath(input_path)
    best_run_json = os.path.join(path, "best_run.json")
    if os.path.isfile(best_run_json):
        with open(best_run_json, "r") as metadata_file:
            metadata = json.load(metadata_file)

        training_log_dir = metadata.get("training_log_dirs")
        if not training_log_dir:
            training_log_dir = metadata.get("training_log_dir")
        if not training_log_dir:
            best_learning_curve = metadata.get("best_learning_curve") or {}
            training_log_dir = best_learning_curve.get("source_log_dirs") or best_learning_curve.get("source_log_dir")
        if not training_log_dir:
            raise ValueError(f"No training_log_dir/source_log_dirs found in {best_run_json}")
        label = _label_for_path(path)
        return _normalize_log_dirs(training_log_dir, best_run_json), label

    learning_curve_metadata = os.path.join(path, "learning_curve_metadata.json")
    if os.path.isfile(learning_curve_metadata):
        with open(learning_curve_metadata, "r") as metadata_file:
            metadata = json.load(metadata_file)
        source_log_dirs = metadata.get("source_log_dirs") or metadata.get("source_log_dir")
        if not source_log_dirs:
            raise ValueError(f"No source_log_dirs found in {learning_curve_metadata}")
        return _normalize_log_dirs(source_log_dirs, learning_curve_metadata), _label_for_path(path)

    summaries_dir = os.path.join(path, "summaries")
    if os.path.isdir(summaries_dir):
        return summaries_dir, _label_for_path(path)

    return path, _label_for_path(path)


def _discover_baseline_runs(
    logs_root: str,
    baselines: list[str],
    task: str | None = None,
    latest_per_family: bool = True,
) -> list[dict[str, Any]]:
    runs_by_task: dict[str, list[tuple[str, Path]]] = collections.defaultdict(list)
    normalized_baselines = list(dict.fromkeys(_normalize_baseline_name(baseline) for baseline in baselines))

    for family in normalized_baselines:
        family_dir = Path(logs_root, family)
        if not family_dir.is_dir():
            continue

        for task_dir in sorted(path for path in family_dir.iterdir() if path.is_dir()):
            if task is not None and task_dir.name != task:
                continue

            for run_dir in sorted(path for path in task_dir.iterdir() if path.is_dir()):
                if (run_dir / "best_run.json").is_file():
                    runs_by_task[task_dir.name].append((family, run_dir))

    if not runs_by_task:
        task_msg = f" for task '{task}'" if task else ""
        raise ValueError(f"No baseline runs found under {os.path.abspath(logs_root)}{task_msg}.")

    selected_task = task
    if selected_task is None:
        ranked_tasks = sorted(
            (
                (task_name, len({family for family, _ in task_runs}), len(task_runs))
                for task_name, task_runs in runs_by_task.items()
            ),
            key=lambda item: (-item[1], -item[2], item[0]),
        )
        best_task, best_family_count, best_run_count = ranked_tasks[0]
        if len(ranked_tasks) > 1 and ranked_tasks[1][1:] == (best_family_count, best_run_count):
            candidate_tasks = ", ".join(
                task_name
                for task_name, family_count, run_count in ranked_tasks
                if (family_count, run_count) == (best_family_count, best_run_count)
            )
            raise ValueError(
                "Multiple tasks have the same discovery coverage. "
                f"Pass --task explicitly. Candidates: {candidate_tasks}"
            )
        selected_task = best_task

    task_runs = runs_by_task.get(selected_task, [])
    if not task_runs:
        raise ValueError(f"No baseline runs found for task '{selected_task}' under {os.path.abspath(logs_root)}.")

    if latest_per_family:
        latest_runs: dict[str, Path] = {}
        for family, run_dir in task_runs:
            previous_run = latest_runs.get(family)
            if previous_run is None or run_dir.name > previous_run.name:
                latest_runs[family] = run_dir
        task_runs = sorted(latest_runs.items(), key=lambda item: item[0])
        return [
            {
                "log_dir": _resolve_log_dir(str(run_dir))[0],
                "label": family,
            }
            for family, run_dir in task_runs
        ]

    return [
        {
            "log_dir": _resolve_log_dir(str(run_dir))[0],
            "label": _label_for_path(str(run_dir)),
        }
        for family, run_dir in sorted(task_runs, key=lambda item: (item[0], item[1].name))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot reward learning curves from RL or baseline run directories.")
    parser.add_argument("inputs", nargs="*", help="Baseline run dirs with best_run.json or direct RL run/log dirs.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for the comparison artifacts. Defaults to recordings/learning_curve_comparisons/<timestamp>.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Learning Curve Comparison",
        help="Title for the comparison plot.",
    )
    parser.add_argument(
        "--logs_root",
        type=str,
        default="logs",
        help="Root directory used for automatic baseline run discovery when no inputs are passed.",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=list(DEFAULT_BASELINES),
        help="Baseline families to include during automatic discovery.",
    )
    parser.add_argument(
        "--task",
        nargs="+",
        default=None,
        help="One or more task names to filter discovered baseline runs, for example Isaac-Cartpole-Direct-v0.",
    )
    parser.add_argument(
        "--all_runs",
        action="store_true",
        help="When using automatic discovery, include all runs instead of only the latest timestamped run per baseline family.",
    )
    parser.add_argument(
        "--metric",
        nargs="+",
        default=["eureka", "oracle", "success"],
        choices=["default", "eureka", "oracle", "success"],
        help="Metric series to compare. Defaults to Eureka reward, oracle reward, and success metric.",
    )
    args = parser.parse_args()

    learning_curve_utils = _load_learning_curve_utils()

    if args.output_dir is None:
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join("recordings", "learning_curve_comparisons", timestamp)
    else:
        output_dir = args.output_dir
    output_dir = os.path.abspath(output_dir)

    task_filters = args.task or [None]

    if args.inputs:
        run_entries: list[dict[str, Any]] = []
        for item in args.inputs:
            log_dir, label = _resolve_log_dir(item)
            run_entries.append({"log_dir": log_dir, "label": label})
        result = learning_curve_utils.export_learning_curve_comparison(
            run_entries=run_entries,
            output_dir=output_dir,
            title=args.title,
            metrics=args.metric,
        )
        if result is None:
            raise RuntimeError(f"No scalar series matching metrics {args.metric} were found in the provided inputs.")

        print(f"[INFO] Compared {len(run_entries)} run(s)")
        print(f"[INFO] Saved comparison plot: {result['plot_path']}")
        print(f"[INFO] Saved comparison CSV: {result['csv_path']}")
        return

    multiple_tasks = len(task_filters) > 1 or task_filters[0] is None
    for task_name in task_filters:
        run_entries = _discover_baseline_runs(
            logs_root=args.logs_root,
            baselines=args.baselines,
            task=task_name,
            latest_per_family=not args.all_runs,
        )

        if task_name is None:
            task_output_dir = output_dir
            task_title = args.title
        elif multiple_tasks:
            task_output_dir = os.path.join(output_dir, _slugify_task_name(task_name))
            task_title = f"{args.title} - {task_name}"
        else:
            task_output_dir = output_dir
            task_title = f"{args.title} - {task_name}"

        result = learning_curve_utils.export_learning_curve_comparison(
            run_entries=run_entries,
            output_dir=task_output_dir,
            title=task_title,
            metrics=args.metric,
        )
        if result is None:
            raise RuntimeError(
                f"No scalar series matching metrics {args.metric} were found for task {task_name or 'auto'}."
            )

        task_label = task_name or "auto"
        print(f"[INFO] Task: {task_label}")
        print(f"[INFO] Compared {len(run_entries)} run(s)")
        print(f"[INFO] Saved comparison plot: {result['plot_path']}")
        print(f"[INFO] Saved comparison CSV: {result['csv_path']}")


if __name__ == "__main__":
    main()
