"""Helpers for exporting TensorBoard learning curves for RL training runs."""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.util import tensor_util

METRIC_MODE_ALIASES = {
    "default": "default",
    "eureka": "eureka",
    "eureka_reward": "eureka",
    "oracle": "oracle",
    "oracle_reward": "oracle",
    "oracle_total_rewards": "oracle",
    "success": "success",
    "success_metric": "success",
}

TASK_SUCCESS_TARGETS = {
    "Isaac-Cartpole-Direct-v0": 1.0,
    "Isaac-Quadcopter-Direct-v0": 0.0,
}

ITERATION_HEADER_RE = re.compile(r"#+\s*Iteration:\s*(\d+)")
RUN_HEADER_RE = re.compile(r"\*+\s*Run:\s*(\d+)")


def load_tensorboard_scalar_series(path: str) -> dict[str, dict[str, list[float]]]:
    """Load TensorBoard scalar-like series including scalar tensors."""
    event_acc = EventAccumulator(path)
    event_acc.Reload()

    data: dict[str, dict[str, list[float]]] = {}
    for tag in event_acc.Tags().get("scalars", []):
        events = event_acc.Scalars(tag)
        data[tag] = {
            "steps": [float(event.step) for event in events],
            "values": [float(event.value) for event in events],
            "wall_times": [float(event.wall_time) for event in events],
        }

    for tag in event_acc.Tags().get("tensors", []):
        events = event_acc.Tensors(tag)
        values: list[float] = []
        steps: list[float] = []
        wall_times: list[float] = []
        for event in events:
            tensor_value = tensor_util.make_ndarray(event.tensor_proto)
            if getattr(tensor_value, "size", 0) != 1:
                continue
            try:
                values.append(float(tensor_value.reshape(-1)[0]))
            except (TypeError, ValueError):
                continue
            steps.append(float(event.step))
            wall_times.append(float(event.wall_time))
        if values:
            data[tag] = {"steps": steps, "values": values, "wall_times": wall_times}

    return data


def select_reward_series_tag(series_by_tag: dict[str, dict[str, list[float]]]) -> str | None:
    """Pick the best reward-like TensorBoard tag for plotting."""
    preferred_tags = [
        "Eureka/eureka_total_rewards",
        "Train/mean_reward",
        "Episode/reward",
        "episode_reward",
        "reward",
    ]
    for tag in preferred_tags:
        if tag in series_by_tag:
            return tag

    reward_candidates = [
        tag
        for tag in series_by_tag
        if "reward" in tag.lower() and "oracle" not in tag.lower() and "loss" not in tag.lower()
    ]
    if reward_candidates:
        reward_candidates.sort(key=lambda item: (item.count("/"), len(item)))
        return reward_candidates[0]
    return None


def normalize_metric_mode(metric: str) -> str:
    """Normalize a metric selector for plotting/export."""
    normalized_metric = metric.strip().lower()
    if normalized_metric not in METRIC_MODE_ALIASES:
        valid_values = ", ".join(sorted(METRIC_MODE_ALIASES))
        raise ValueError(f"Unsupported metric '{metric}'. Expected one of: {valid_values}")
    return METRIC_MODE_ALIASES[normalized_metric]


def select_plot_series_tag(series_by_tag: dict[str, dict[str, list[float]]], metric: str = "default") -> str | None:
    """Pick the scalar series to plot for a requested metric mode."""
    metric_mode = normalize_metric_mode(metric)
    if metric_mode == "default":
        return select_reward_series_tag(series_by_tag)
    if metric_mode == "success":
        return select_success_metric_tag(series_by_tag)

    preferred_tags_by_metric = {
        "eureka": [
            "Eureka/eureka_total_rewards",
            "eureka_total_rewards",
        ],
        "oracle": [
            "Eureka/oracle_total_rewards",
            "oracle_total_rewards",
        ],
    }
    for tag in preferred_tags_by_metric[metric_mode]:
        if tag in series_by_tag:
            return tag
    return None


def select_success_metric_tag(series_by_tag: dict[str, dict[str, list[float]]]) -> str | None:
    """Pick a success metric tag when available."""
    preferred_tags = [
        "Eureka/success_metric",
        "success_metric",
    ]
    for tag in preferred_tags:
        if tag in series_by_tag:
            return tag
    return None


def select_episode_length_tag(series_by_tag: dict[str, dict[str, list[float]]]) -> str | None:
    """Pick a mean episode length tag when available."""
    preferred_tags = [
        "Train/mean_episode_length",
        "Episode/length",
        "episode_length",
    ]
    for tag in preferred_tags:
        if tag in series_by_tag:
            return tag
    return None


def resolve_checkpoint_path(run_dir: str | None) -> str | None:
    """Pick the latest checkpoint file from a training run directory."""
    if not run_dir or not os.path.isdir(run_dir):
        return None

    patterns = [
        os.path.join(run_dir, "model_*.pt"),
        os.path.join(run_dir, "*.pth"),
        os.path.join(run_dir, "nn", "*.pth"),
        os.path.join(run_dir, "**", "model_*.pt"),
        os.path.join(run_dir, "**", "*.pth"),
    ]

    checkpoint_paths: list[str] = []
    for pattern in patterns:
        checkpoint_paths.extend(glob.glob(pattern, recursive=True))

    checkpoint_paths = sorted(set(path for path in checkpoint_paths if os.path.isfile(path)))
    if not checkpoint_paths:
        return None

    def _checkpoint_sort_key(path: str) -> tuple[int, float]:
        filename = os.path.basename(path)
        digits = "".join(ch for ch in filename if ch.isdigit())
        step = int(digits) if digits else -1
        return step, os.path.getmtime(path)

    return max(checkpoint_paths, key=_checkpoint_sort_key)


def _infer_baseline_run_context(log_dir: str) -> dict[str, str] | None:
    """Infer logs/<family>/<task>/<timestamp> from a nested RL run directory."""
    normalized_path = os.path.abspath(log_dir)
    path_parts = normalized_path.split(os.sep)
    try:
        logs_index = path_parts.index("logs")
    except ValueError:
        return None
    if len(path_parts) <= logs_index + 3:
        return None
    family = path_parts[logs_index + 1]
    task = path_parts[logs_index + 2]
    run_name = path_parts[logs_index + 3]
    run_root = os.path.join(*path_parts[: logs_index + 4])
    if normalized_path.startswith(os.sep):
        run_root = os.sep + run_root
    return {
        "family": family,
        "task": task,
        "run_name": run_name,
        "run_root": os.path.abspath(run_root),
    }


def _parse_iteration_statuses(iteration_log_path: str) -> dict[tuple[int, int], bool]:
    """Parse per-iteration, per-run success/failure flags from a baseline log file."""
    if not os.path.isfile(iteration_log_path):
        return {}

    statuses: dict[tuple[int, int], bool] = {}
    current_iteration: int | None = None
    current_run: int | None = None
    with open(iteration_log_path, "r") as log_file:
        for line in log_file:
            iteration_match = ITERATION_HEADER_RE.search(line)
            if iteration_match:
                current_iteration = int(iteration_match.group(1))
                continue

            run_match = RUN_HEADER_RE.search(line)
            if run_match:
                current_run = int(run_match.group(1))
                continue

            if current_iteration is None or current_run is None:
                continue

            if "Training successful" in line:
                statuses[(current_iteration, current_run)] = True
            elif "Training failed" in line:
                statuses[(current_iteration, current_run)] = False

    return statuses


def _load_iteration_success_from_tensorboard(
    run_root: str,
    iteration_log_name: str,
    success_metric_target: float,
) -> dict[str, Any] | None:
    """Load best-per-iteration success metric for baselines that log root TensorBoard scalars."""
    statuses = _parse_iteration_statuses(os.path.join(run_root, iteration_log_name))
    if not statuses:
        return None

    series_by_tag = load_tensorboard_scalar_series(run_root)
    run_success_tags: dict[int, dict[str, dict[str, list[float]]]] = {}
    for tag, series in series_by_tag.items():
        success_match = re.fullmatch(r"Run_(\d+)/success_metric", tag)
        if success_match:
            run_success_tags.setdefault(int(success_match.group(1)), {})["success"] = series
            continue
        stderr_match = re.fullmatch(r"Run_(\d+)/success_metric_stderr", tag)
        if stderr_match:
            run_success_tags.setdefault(int(stderr_match.group(1)), {})["stderr"] = series

    if not run_success_tags:
        return None

    max_iteration = max(iteration for iteration, _ in statuses)
    rows: list[dict[str, Any]] = []
    for iteration in range(max_iteration + 1):
        iteration_candidates: list[dict[str, Any]] = []
        attempted_runs = 0
        successful_runs = 0

        for (iter_idx, run_idx), was_successful in statuses.items():
            if iter_idx != iteration:
                continue
            attempted_runs += 1
            if not was_successful:
                continue

            run_series = run_success_tags.get(run_idx, {})
            success_series = run_series.get("success")
            if not success_series:
                continue

            success_by_step = {
                int(step): float(value)
                for step, value in zip(success_series["steps"], success_series["values"])
            }
            if iteration not in success_by_step:
                continue

            successful_runs += 1
            stderr_value = None
            stderr_series = run_series.get("stderr")
            if stderr_series:
                stderr_by_step = {
                    int(step): float(value)
                    for step, value in zip(stderr_series["steps"], stderr_series["values"])
                }
                stderr_value = stderr_by_step.get(iteration)

            iteration_candidates.append(
                {
                    "run_idx": run_idx,
                    "success_metric": success_by_step[iteration],
                    "success_metric_stderr": stderr_value,
                }
            )

        selected_candidate = None
        if iteration_candidates:
            selected_candidate = min(
                iteration_candidates,
                key=lambda item: abs(float(item["success_metric"]) - success_metric_target),
            )

        rows.append(
            {
                "iteration": iteration,
                "success_metric": selected_candidate["success_metric"] if selected_candidate else None,
                "success_metric_stderr": selected_candidate["success_metric_stderr"] if selected_candidate else None,
                "selected_run_idx": selected_candidate["run_idx"] if selected_candidate else None,
                "attempted_runs": attempted_runs,
                "successful_runs": successful_runs,
            }
        )

    return {"rows": rows, "num_iterations": max_iteration + 1}


def _load_iteration_success_from_revolve_full_database(
    run_root: str,
    success_metric_target: float,
) -> dict[str, Any] | None:
    """Load best-per-generation success metric for REvolve full from database fitness JSON files."""
    fitness_paths = sorted(glob.glob(os.path.join(run_root, "database", "island_*", "fitness_scores", "*.txt")))
    if not fitness_paths:
        return None

    generations: dict[int, list[dict[str, Any]]] = {}
    max_generation = -1
    for fitness_path in fitness_paths:
        filename = os.path.splitext(os.path.basename(fitness_path))[0]
        try:
            generation_str, counter_str = filename.split("_", 1)
            generation = int(generation_str)
            counter = int(counter_str)
        except ValueError:
            continue

        max_generation = max(max_generation, generation)
        with open(fitness_path, "r") as fitness_file:
            try:
                record = json.load(fitness_file)
            except json.JSONDecodeError:
                continue

        generations.setdefault(generation, []).append(
            {
                "counter": counter,
                "success": bool(record.get("success")),
                "success_metric": record.get("success_metric"),
                "success_metric_stderr": record.get("success_metric_stderr"),
            }
        )

    if max_generation < 0:
        return None

    rows: list[dict[str, Any]] = []
    for generation in range(max_generation + 1):
        generation_candidates = generations.get(generation, [])
        successful_candidates = [
            candidate
            for candidate in generation_candidates
            if candidate.get("success") and candidate.get("success_metric") is not None
        ]

        selected_candidate = None
        if successful_candidates:
            selected_candidate = min(
                successful_candidates,
                key=lambda item: abs(float(item["success_metric"]) - success_metric_target),
            )

        rows.append(
            {
                "iteration": generation,
                "success_metric": selected_candidate["success_metric"] if selected_candidate else None,
                "success_metric_stderr": selected_candidate["success_metric_stderr"] if selected_candidate else None,
                "selected_run_idx": selected_candidate["counter"] if selected_candidate else None,
                "attempted_runs": len(generation_candidates),
                "successful_runs": len(successful_candidates),
            }
        )

    return {"rows": rows, "num_iterations": max_generation + 1}


def load_outer_iteration_success_series(log_dirs: list[str] | tuple[str, ...] | str) -> dict[str, Any] | None:
    """Load best outer-iteration success metric series for a baseline run."""
    if isinstance(log_dirs, str):
        normalized_log_dirs = [os.path.abspath(log_dirs)]
    else:
        normalized_log_dirs = [os.path.abspath(str(path)) for path in log_dirs if path]
    if not normalized_log_dirs:
        return None

    context = _infer_baseline_run_context(normalized_log_dirs[0])
    if context is None:
        return None

    success_metric_target = TASK_SUCCESS_TARGETS.get(context["task"])
    if success_metric_target is None:
        return None

    family = context["family"]
    run_root = context["run_root"]
    if family in {"eureka", "tacreka_sr", "tacreka_preference", "tacreka_ranking"}:
        result = _load_iteration_success_from_tensorboard(
            run_root=run_root,
            iteration_log_name="eureka_iterations.txt",
            success_metric_target=success_metric_target,
        )
    elif family == "revolve":
        result = _load_iteration_success_from_tensorboard(
            run_root=run_root,
            iteration_log_name="revolve_iterations.txt",
            success_metric_target=success_metric_target,
        )
    elif family == "revolve_full":
        result = _load_iteration_success_from_revolve_full_database(
            run_root=run_root,
            success_metric_target=success_metric_target,
        )
    else:
        return None

    if result is None:
        return None

    result["family"] = family
    result["task"] = context["task"]
    result["run_root"] = run_root
    result["success_metric_target"] = success_metric_target
    return result


def _mean_and_stderr(values: list[float]) -> tuple[float | None, float | None]:
    """Compute the mean and standard error for a sequence of scalars."""
    if not values:
        return None, None
    if len(values) == 1:
        return float(values[0]), 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    stderr = math.sqrt(variance) / math.sqrt(len(values))
    return float(mean), float(stderr)


def export_learning_curve_artifacts(
    log_dir: str | list[str] | tuple[str, ...],
    output_dir: str | None = None,
    run_name: str | None = None,
    metric: str = "default",
) -> dict[str, Any] | None:
    """Write a training-step learning curve plot plus CSV metadata."""
    if isinstance(log_dir, (list, tuple)):
        normalized_log_dirs = [os.path.abspath(path) for path in log_dir]
    else:
        normalized_log_dirs = [os.path.abspath(log_dir)]

    series_by_run = [load_tensorboard_scalar_series(path) for path in normalized_log_dirs]
    metric_mode = normalize_metric_mode(metric)
    reward_tag = None
    for series_by_tag in series_by_run:
        reward_tag = select_plot_series_tag(series_by_tag, metric=metric_mode)
        if reward_tag is not None:
            break
    if reward_tag is None:
        return None

    first_series = next(series_by_tag for series_by_tag in series_by_run if reward_tag in series_by_tag)
    success_tag = select_success_metric_tag(first_series)
    episode_length_tag = select_episode_length_tag(first_series)
    output_dir = os.path.abspath(output_dir or normalized_log_dirs[0])
    os.makedirs(output_dir, exist_ok=True)

    reward_values_by_step: dict[float, list[float]] = {}
    success_values_by_step: dict[float, list[float]] = {}
    episode_length_values_by_step: dict[float, list[float]] = {}

    for series_by_tag in series_by_run:
        current_reward_tag = select_plot_series_tag(series_by_tag, metric=metric_mode)
        if current_reward_tag is None or current_reward_tag not in series_by_tag:
            continue

        reward_series = series_by_tag[current_reward_tag]
        for step, value in zip(reward_series["steps"], reward_series["values"]):
            reward_values_by_step.setdefault(step, []).append(float(value))

        if success_tag and success_tag in series_by_tag:
            success_step_to_value = dict(zip(series_by_tag[success_tag]["steps"], series_by_tag[success_tag]["values"]))
            for step in reward_series["steps"]:
                if step in success_step_to_value:
                    success_values_by_step.setdefault(step, []).append(float(success_step_to_value[step]))

        if episode_length_tag and episode_length_tag in series_by_tag:
            episode_step_to_value = dict(
                zip(series_by_tag[episode_length_tag]["steps"], series_by_tag[episode_length_tag]["values"])
            )
            for step in reward_series["steps"]:
                if step in episode_step_to_value:
                    episode_length_values_by_step.setdefault(step, []).append(float(episode_step_to_value[step]))

    reward_steps = sorted(reward_values_by_step)
    reward_values = []
    reward_stderr_values = []
    reward_count_values = []
    success_values: list[float | None] = []
    success_stderr_values: list[float | None] = []
    success_count_values: list[int] = []
    episode_length_values: list[float | None] = []
    episode_length_stderr_values: list[float | None] = []
    episode_length_count_values: list[int] = []

    for step in reward_steps:
        reward_mean, reward_stderr = _mean_and_stderr(reward_values_by_step.get(step, []))
        reward_values.append(reward_mean if reward_mean is not None else 0.0)
        reward_stderr_values.append(reward_stderr if reward_stderr is not None else 0.0)
        reward_count_values.append(len(reward_values_by_step.get(step, [])))

        success_mean, success_stderr = _mean_and_stderr(success_values_by_step.get(step, []))
        success_values.append(success_mean)
        success_stderr_values.append(success_stderr)
        success_count_values.append(len(success_values_by_step.get(step, [])))

        episode_mean, episode_stderr = _mean_and_stderr(episode_length_values_by_step.get(step, []))
        episode_length_values.append(episode_mean)
        episode_length_stderr_values.append(episode_stderr)
        episode_length_count_values.append(len(episode_length_values_by_step.get(step, [])))

    logged_indices = list(range(len(reward_values)))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_title = run_name or os.path.basename(normalized_log_dirs[0])
    figure, axis = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)

    axis.plot(reward_steps, reward_values, color="#1f77b4", linewidth=2)
    if len(normalized_log_dirs) > 1:
        lower = [value - stderr for value, stderr in zip(reward_values, reward_stderr_values)]
        upper = [value + stderr for value, stderr in zip(reward_values, reward_stderr_values)]
        axis.fill_between(reward_steps, lower, upper, color="#1f77b4", alpha=0.2)
    axis.set_title(f"{figure_title}\n{reward_tag} vs Training Step")
    axis.set_xlabel("Training Step")
    axis.set_ylabel(reward_tag)
    axis.grid(True, alpha=0.25)

    plot_path = os.path.join(output_dir, "learning_curves.png")
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)

    csv_path = os.path.join(output_dir, "learning_curve_data.csv")
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "logged_index",
                "training_step",
                "reward",
                "reward_stderr",
                "reward_count",
                "success_metric",
                "success_metric_stderr",
                "success_metric_count",
                "mean_episode_length",
                "mean_episode_length_stderr",
                "mean_episode_length_count",
            ],
        )
        writer.writeheader()
        for (
            idx,
            step,
            reward,
            reward_stderr,
            reward_count,
            success_value,
            success_stderr,
            success_count,
            episode_length_value,
            episode_length_stderr,
            episode_length_count,
        ) in zip(
            logged_indices,
            reward_steps,
            reward_values,
            reward_stderr_values,
            reward_count_values,
            success_values,
            success_stderr_values,
            success_count_values,
            episode_length_values,
            episode_length_stderr_values,
            episode_length_count_values,
        ):
            writer.writerow(
                {
                    "logged_index": idx,
                    "training_step": step,
                    "reward": reward,
                    "reward_stderr": reward_stderr,
                    "reward_count": reward_count,
                    "success_metric": success_value if success_value is not None else "",
                    "success_metric_stderr": success_stderr if success_stderr is not None else "",
                    "success_metric_count": success_count,
                    "mean_episode_length": episode_length_value if episode_length_value is not None else "",
                    "mean_episode_length_stderr": episode_length_stderr if episode_length_stderr is not None else "",
                    "mean_episode_length_count": episode_length_count,
                }
            )

    metadata = {
        "source_log_dir": normalized_log_dirs[0],
        "source_log_dirs": normalized_log_dirs,
        "metric_mode": metric_mode,
        "series_tag": reward_tag,
        "reward_tag": reward_tag,
        "success_metric_tag": success_tag,
        "mean_episode_length_tag": episode_length_tag,
        "num_runs": len(normalized_log_dirs),
        "num_points": len(reward_values),
        "max_reward": max(reward_values) if reward_values else None,
        "final_reward": reward_values[-1] if reward_values else None,
        "plot_path": plot_path,
        "csv_path": csv_path,
    }
    metadata_path = os.path.join(output_dir, "learning_curve_metadata.json")
    with open(metadata_path, "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    metadata["metadata_path"] = metadata_path
    return metadata


def export_learning_curve_comparison(
    run_entries: list[dict[str, Any]],
    output_dir: str,
    title: str = "Learning Curve Comparison",
    metrics: str | list[str] | tuple[str, ...] | None = None,
) -> dict[str, str] | None:
    """Create overlay plots for multiple training runs."""
    if not run_entries:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.transforms as mtransforms

    if metrics is None:
        requested_metrics = ["eureka", "oracle", "success"]
    elif isinstance(metrics, str):
        requested_metrics = [metrics]
    else:
        requested_metrics = list(metrics)

    metric_modes = list(dict.fromkeys(normalize_metric_mode(metric) for metric in requested_metrics))
    metric_titles = {
        "default": "Default Reward",
        "eureka": "Eureka Reward",
        "oracle": "Oracle Reward",
        "success": "Success Metric",
    }

    os.makedirs(output_dir, exist_ok=True)
    comparison_rows: list[dict[str, Any]] = []
    iteration_rows: list[dict[str, Any]] = []
    iteration_summaries: dict[str, dict[str, Any]] = {}
    for entry in run_entries:
        log_dir_input = entry["log_dir"]
        if isinstance(log_dir_input, (list, tuple)):
            normalized_log_dirs = [os.path.abspath(str(path)) for path in log_dir_input]
        else:
            normalized_log_dirs = [os.path.abspath(str(log_dir_input))]
        iteration_summary = load_outer_iteration_success_series(normalized_log_dirs)
        if iteration_summary is not None:
            iteration_summaries[entry["label"]] = iteration_summary

    figure, axes = plt.subplots(1, len(metric_modes), figsize=(8 * len(metric_modes), 5), constrained_layout=True)
    if len(metric_modes) == 1:
        axes = [axes]

    plotted_any = False
    for axis, metric_mode in zip(axes, metric_modes):
        y_axis_label = "Reward"
        plotted_metric = 0

        for entry in run_entries:
            label = entry["label"]
            log_dir_input = entry["log_dir"]
            if isinstance(log_dir_input, (list, tuple)):
                normalized_log_dirs = [os.path.abspath(str(path)) for path in log_dir_input]
            else:
                normalized_log_dirs = [os.path.abspath(str(log_dir_input))]

            series_by_run = [load_tensorboard_scalar_series(path) for path in normalized_log_dirs]
            reward_tag = None
            for series_by_tag in series_by_run:
                reward_tag = select_plot_series_tag(series_by_tag, metric=metric_mode)
                if reward_tag is not None:
                    break
            if reward_tag is None:
                continue

            reward_values_by_step: dict[float, list[float]] = {}
            for series_by_tag in series_by_run:
                current_reward_tag = select_plot_series_tag(series_by_tag, metric=metric_mode)
                if current_reward_tag is None or current_reward_tag not in series_by_tag:
                    continue
                reward_series = series_by_tag[current_reward_tag]
                for step, value in zip(reward_series["steps"], reward_series["values"]):
                    reward_values_by_step.setdefault(float(step), []).append(float(value))

            if not reward_values_by_step:
                continue

            reward_steps = sorted(reward_values_by_step)
            reward_values: list[float] = []
            reward_stderr_values: list[float] = []
            reward_count_values: list[int] = []
            for step in reward_steps:
                reward_mean, reward_stderr = _mean_and_stderr(reward_values_by_step[step])
                reward_values.append(reward_mean if reward_mean is not None else 0.0)
                reward_stderr_values.append(reward_stderr if reward_stderr is not None else 0.0)
                reward_count_values.append(len(reward_values_by_step[step]))

            logged_indices = list(range(len(reward_values)))
            if plotted_metric == 0:
                y_axis_label = reward_tag

            iteration_count = iteration_summaries.get(label, {}).get("num_iterations")
            plot_label = f"{label} ({iteration_count} iters)" if iteration_count is not None else label
            axis.plot(reward_steps, reward_values, linewidth=2, label=plot_label)
            if len(normalized_log_dirs) > 1:
                lower = [value - stderr for value, stderr in zip(reward_values, reward_stderr_values)]
                upper = [value + stderr for value, stderr in zip(reward_values, reward_stderr_values)]
                axis.fill_between(reward_steps, lower, upper, alpha=0.18)
            plotted_metric += 1
            plotted_any = True

            source_log_dir = normalized_log_dirs[0]
            source_log_dirs = json.dumps(normalized_log_dirs)
            for idx, step, reward, reward_stderr, reward_count in zip(
                logged_indices, reward_steps, reward_values, reward_stderr_values, reward_count_values
            ):
                comparison_rows.append(
                    {
                        "metric_mode": metric_mode,
                        "label": label,
                        "log_dir": source_log_dir,
                        "source_log_dirs": source_log_dirs,
                        "logged_index": idx,
                        "training_step": step,
                        "reward": reward,
                        "reward_stderr": reward_stderr,
                        "reward_count": reward_count,
                        "series_value": reward,
                        "series_stderr": reward_stderr,
                        "series_count": reward_count,
                        "reward_tag": reward_tag,
                        "series_tag": reward_tag,
                    }
                )

        if plotted_metric == 0:
            axis.set_axis_off()
            axis.text(0.5, 0.5, f"No data for {metric_mode}", ha="center", va="center")
            continue

        axis.set_title(f"{title}\n{metric_titles[metric_mode]} vs Training Step")
        axis.set_xlabel("Training Step")
        axis.set_ylabel(y_axis_label)
        axis.grid(True, alpha=0.25)
        axis.legend()

    if not plotted_any:
        plt.close(figure)
        return None

    plot_path = os.path.join(output_dir, "learning_curve_comparison.png")
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)

    iteration_plot_path = None
    iteration_csv_path = None
    if iteration_summaries:
        iteration_figure, iteration_axis = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)
        plotted_iteration_series = False
        missing_iteration_annotations = []
        all_iteration_success_values = []
        annotation_transform = mtransforms.blended_transform_factory(
            iteration_axis.transData,
            iteration_axis.transAxes,
        )

        for series_index, entry in enumerate(run_entries):
            label = entry["label"]
            summary = iteration_summaries.get(label)
            if summary is None:
                continue

            rows = summary["rows"]
            if not rows:
                continue

            iteration_values = [row["iteration"] for row in rows]
            success_values = [
                float(row["success_metric"]) if row["success_metric"] is not None else float("nan")
                for row in rows
            ]
            stderr_values = [
                float(row["success_metric_stderr"]) if row["success_metric_stderr"] is not None else 0.0
                for row in rows
            ]
            valid_success_values = [value for value in success_values if not math.isnan(value)]
            iteration_count = summary.get("num_iterations", len(rows))
            plot_label = f"{label} ({iteration_count} iters)"

            line, = iteration_axis.plot(
                iteration_values,
                success_values,
                linewidth=2,
                marker="o",
                markersize=4,
                label=plot_label,
            )
            for row in rows:
                if row["success_metric"] is None:
                    missing_iteration_annotations.append(
                        {
                            "iteration": row["iteration"],
                            "label": label,
                            "color": line.get_color(),
                            "series_index": series_index,
                        }
                    )
            if valid_success_values:
                all_iteration_success_values.extend(valid_success_values)
                lower = [
                    (value - stderr) if not math.isnan(value) else float("nan")
                    for value, stderr in zip(success_values, stderr_values)
                ]
                upper = [
                    (value + stderr) if not math.isnan(value) else float("nan")
                    for value, stderr in zip(success_values, stderr_values)
                ]
                iteration_axis.fill_between(iteration_values, lower, upper, color=line.get_color(), alpha=0.18)
                plotted_iteration_series = True

            source_log_dir = entry["log_dir"][0] if isinstance(entry["log_dir"], (list, tuple)) else entry["log_dir"]
            for row in rows:
                iteration_rows.append(
                    {
                        "label": label,
                        "run_root": summary["run_root"],
                        "task": summary["task"],
                        "family": summary["family"],
                        "iteration": row["iteration"],
                        "success_metric": row["success_metric"],
                        "success_metric_stderr": row["success_metric_stderr"],
                        "attempted_runs": row["attempted_runs"],
                        "successful_runs": row["successful_runs"],
                        "selected_run_idx": row["selected_run_idx"],
                        "iteration_count": iteration_count,
                        "success_metric_target": summary["success_metric_target"],
                        "source_log_dir": source_log_dir,
                    }
                )

        if plotted_iteration_series:
            target = next(iter(iteration_summaries.values())).get("success_metric_target")
            if target is not None:
                iteration_axis.axhline(float(target), color="black", linestyle="--", linewidth=1, alpha=0.5)
            if target is not None:
                all_iteration_success_values.append(float(target))
            if missing_iteration_annotations:
                iteration_slot_counts = {}
                for annotation in missing_iteration_annotations:
                    iteration_value = annotation["iteration"]
                    slot = iteration_slot_counts.get(iteration_value, 0)
                    iteration_slot_counts[iteration_value] = slot + 1
                    marker_y = 0.04 + slot * 0.07
                    text_y = 0.06 + slot * 0.07
                    iteration_axis.scatter(
                        [iteration_value],
                        [marker_y],
                        marker="x",
                        color=annotation["color"],
                        s=32,
                        linewidths=1.4,
                        zorder=6,
                        transform=annotation_transform,
                        clip_on=False,
                    )
                    iteration_axis.text(
                        iteration_value,
                        text_y,
                        "no data",
                        color=annotation["color"],
                        fontsize=8,
                        ha="center",
                        va="bottom",
                        rotation=20,
                        transform=annotation_transform,
                        clip_on=False,
                    )
            iteration_axis.set_title(f"{title}\nBest Success Metric vs Outer Iteration")
            iteration_axis.set_xlabel("Outer Iteration")
            iteration_axis.set_ylabel("Success Metric")
            iteration_axis.grid(True, alpha=0.25)
            iteration_axis.legend()
            iteration_plot_path = os.path.join(output_dir, "iteration_success_comparison.png")
            iteration_figure.savefig(iteration_plot_path, dpi=180)
        plt.close(iteration_figure)

        if iteration_rows:
            iteration_csv_path = os.path.join(output_dir, "iteration_success_comparison.csv")
            with open(iteration_csv_path, "w", newline="") as iteration_csv_file:
                writer = csv.DictWriter(
                    iteration_csv_file,
                    fieldnames=[
                        "label",
                        "run_root",
                        "task",
                        "family",
                        "iteration",
                        "success_metric",
                        "success_metric_stderr",
                        "attempted_runs",
                        "successful_runs",
                        "selected_run_idx",
                        "iteration_count",
                        "success_metric_target",
                        "source_log_dir",
                    ],
                )
                writer.writeheader()
                writer.writerows(iteration_rows)

    csv_path = os.path.join(output_dir, "learning_curve_comparison.csv")
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "metric_mode",
                "label",
                "log_dir",
                "source_log_dirs",
                "logged_index",
                "training_step",
                "reward",
                "reward_stderr",
                "reward_count",
                "series_value",
                "series_stderr",
                "series_count",
                "reward_tag",
                "series_tag",
            ],
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    result = {"plot_path": plot_path, "csv_path": csv_path}
    if iteration_plot_path is not None:
        result["iteration_plot_path"] = iteration_plot_path
    if iteration_csv_path is not None:
        result["iteration_csv_path"] = iteration_csv_path
    return result
