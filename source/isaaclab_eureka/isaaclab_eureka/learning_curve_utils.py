"""Helpers for exporting TensorBoard learning curves for RL training runs."""

from __future__ import annotations

import csv
import glob
import json
import os
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
}


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
            values.append(float(tensor_value.reshape(-1)[0]))
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


def export_learning_curve_artifacts(
    log_dir: str,
    output_dir: str | None = None,
    run_name: str | None = None,
    metric: str = "default",
) -> dict[str, Any] | None:
    """Write a training-step learning curve plot plus CSV metadata."""
    series_by_tag = load_tensorboard_scalar_series(log_dir)
    metric_mode = normalize_metric_mode(metric)
    reward_tag = select_plot_series_tag(series_by_tag, metric=metric_mode)
    if reward_tag is None:
        return None

    success_tag = select_success_metric_tag(series_by_tag)
    episode_length_tag = select_episode_length_tag(series_by_tag)
    reward_series = series_by_tag[reward_tag]
    output_dir = os.path.abspath(output_dir or log_dir)
    os.makedirs(output_dir, exist_ok=True)

    reward_steps = reward_series["steps"]
    reward_values = reward_series["values"]
    logged_indices = list(range(len(reward_values)))

    success_values: list[float | None]
    if success_tag and success_tag in series_by_tag:
        success_step_to_value = dict(zip(series_by_tag[success_tag]["steps"], series_by_tag[success_tag]["values"]))
        success_values = [success_step_to_value.get(step) for step in reward_steps]
    else:
        success_values = [None] * len(reward_values)

    episode_length_values: list[float | None]
    if episode_length_tag and episode_length_tag in series_by_tag:
        episode_step_to_value = dict(
            zip(series_by_tag[episode_length_tag]["steps"], series_by_tag[episode_length_tag]["values"])
        )
        episode_length_values = [episode_step_to_value.get(step) for step in reward_steps]
    else:
        episode_length_values = [None] * len(reward_values)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_title = run_name or os.path.basename(os.path.abspath(log_dir))
    figure, axis = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)

    axis.plot(reward_steps, reward_values, color="#1f77b4", linewidth=2)
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
                "success_metric",
                "mean_episode_length",
            ],
        )
        writer.writeheader()
        for idx, step, reward, success_value, episode_length_value in zip(
            logged_indices, reward_steps, reward_values, success_values, episode_length_values
        ):
            writer.writerow(
                {
                    "logged_index": idx,
                    "training_step": step,
                    "reward": reward,
                    "success_metric": success_value if success_value is not None else "",
                    "mean_episode_length": episode_length_value if episode_length_value is not None else "",
                }
            )

    metadata = {
        "source_log_dir": os.path.abspath(log_dir),
        "metric_mode": metric_mode,
        "series_tag": reward_tag,
        "reward_tag": reward_tag,
        "success_metric_tag": success_tag,
        "mean_episode_length_tag": episode_length_tag,
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
    run_entries: list[dict[str, str]],
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

    if metrics is None:
        requested_metrics = ["default", "oracle"]
    elif isinstance(metrics, str):
        requested_metrics = [metrics]
    else:
        requested_metrics = list(metrics)

    metric_modes = list(dict.fromkeys(normalize_metric_mode(metric) for metric in requested_metrics))
    metric_titles = {
        "default": "Default Reward",
        "eureka": "Eureka Reward",
        "oracle": "Oracle Reward",
    }

    os.makedirs(output_dir, exist_ok=True)
    comparison_rows: list[dict[str, Any]] = []
    figure, axes = plt.subplots(1, len(metric_modes), figsize=(8 * len(metric_modes), 5), constrained_layout=True)
    if len(metric_modes) == 1:
        axes = [axes]

    plotted_any = False
    for axis, metric_mode in zip(axes, metric_modes):
        y_axis_label = "Reward"
        plotted_metric = 0

        for entry in run_entries:
            label = entry["label"]
            log_dir = entry["log_dir"]
            series_by_tag = load_tensorboard_scalar_series(log_dir)
            reward_tag = select_plot_series_tag(series_by_tag, metric=metric_mode)
            if reward_tag is None:
                continue

            reward_series = series_by_tag[reward_tag]
            reward_steps = reward_series["steps"]
            reward_values = reward_series["values"]
            logged_indices = list(range(len(reward_values)))
            if plotted_metric == 0:
                y_axis_label = reward_tag

            axis.plot(reward_steps, reward_values, linewidth=2, label=label)
            plotted_metric += 1
            plotted_any = True

            for idx, step, reward in zip(logged_indices, reward_steps, reward_values):
                comparison_rows.append(
                    {
                        "metric_mode": metric_mode,
                        "label": label,
                        "log_dir": os.path.abspath(log_dir),
                        "logged_index": idx,
                        "training_step": step,
                        "reward": reward,
                        "reward_tag": reward_tag,
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

    csv_path = os.path.join(output_dir, "learning_curve_comparison.csv")
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["metric_mode", "label", "log_dir", "logged_index", "training_step", "reward", "reward_tag"],
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    return {"plot_path": plot_path, "csv_path": csv_path}
