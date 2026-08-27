# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

import math
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

import GPUtil
from isaaclab_eureka.learning_curve_utils import load_tensorboard_scalar_series


def load_tensorboard_logs(path: str):
    """Load tensorboard logs from a given path.

    Args:
        path: The path to the tensorboard logs.

    Returns:
        A dictionary with the tags and their respective values.
    """
    data = defaultdict(list)
    for tag, series in load_tensorboard_scalar_series(path).items():
        data[tag].extend(series["values"])

    return data


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


def _aggregate_scalar_sequences(metric_sequences: list[list[float]]) -> tuple[list[float], list[float]]:
    """Aggregate scalar sequences with the same semantic tag across seeds by index."""
    if not metric_sequences:
        return [], []

    max_length = max(len(sequence) for sequence in metric_sequences)
    mean_series: list[float] = []
    stderr_series: list[float] = []
    for index in range(max_length):
        values = [sequence[index] for sequence in metric_sequences if index < len(sequence)]
        mean_value, stderr_value = _mean_and_stderr(values)
        if mean_value is None or stderr_value is None:
            continue
        mean_series.append(mean_value)
        stderr_series.append(stderr_value)
    return mean_series, stderr_series


def _compute_reward_correlation(data: dict[str, list[float]]) -> float:
    """Compute Pearson correlation between Eureka and oracle total rewards for one seed."""
    eureka_values = [float(value) for value in data.get("Eureka/eureka_total_rewards", [])[2:]]
    oracle_values = [float(value) for value in data.get("Eureka/oracle_total_rewards", [])[2:]]
    if len(eureka_values) < 2 or len(oracle_values) < 2:
        return 0.0
    paired_length = min(len(eureka_values), len(oracle_values))
    if paired_length < 2:
        return 0.0

    eureka_values = eureka_values[:paired_length]
    oracle_values = oracle_values[:paired_length]
    mean_x = sum(eureka_values) / paired_length
    mean_y = sum(oracle_values) / paired_length
    centered_x = [value - mean_x for value in eureka_values]
    centered_y = [value - mean_y for value in oracle_values]
    numerator = sum(x * y for x, y in zip(centered_x, centered_y))
    denom_x = math.sqrt(sum(x * x for x in centered_x))
    denom_y = math.sqrt(sum(y * y for y in centered_y))
    if denom_x == 0.0 or denom_y == 0.0:
        return 0.0
    return float(numerator / (denom_x * denom_y))


def summarize_tensorboard_candidate(
    log_dirs: str | list[str] | tuple[str, ...],
    feedback_subsampling: int,
    success_metric_target: float,
) -> dict:
    """Aggregate one candidate across one or more seed runs."""
    if isinstance(log_dirs, (list, tuple)):
        normalized_log_dirs = [os.path.abspath(log_dir) for log_dir in log_dirs]
    else:
        normalized_log_dirs = [os.path.abspath(log_dirs)]

    metrics_by_tag: dict[str, list[list[float]]] = defaultdict(list)
    seed_summaries: list[dict] = []
    has_success_metric = False
    subsampling = max(1, int(feedback_subsampling))

    for seed_index, log_dir in enumerate(normalized_log_dirs):
        data = load_tensorboard_logs(log_dir)
        success_metric_value = None

        for metric_name, metric_data in data.items():
            if "Eureka/" not in metric_name:
                continue
            trimmed_metric = [float(value) for value in metric_data[2:]]
            metrics_by_tag[metric_name].append(trimmed_metric)

            if metric_name.endswith("Eureka/success_metric"):
                has_success_metric = True
                if trimmed_metric:
                    tail_count = max(1, math.ceil(len(trimmed_metric) * 0.1))
                    success_metric_value = float(sum(trimmed_metric[-tail_count:]) / tail_count)

        seed_summaries.append(
            {
                "seed_index": seed_index,
                "log_dir": log_dir,
                "success_metric": success_metric_value,
                "rewards_correlation": _compute_reward_correlation(data),
            }
        )

    success_metric_values = [
        float(summary["success_metric"])
        for summary in seed_summaries
        if summary.get("success_metric") is not None
    ]
    rewards_correlation_values = [float(summary["rewards_correlation"]) for summary in seed_summaries]

    success_metric_mean, success_metric_stderr = _mean_and_stderr(success_metric_values)
    rewards_correlation_mean, rewards_correlation_stderr = _mean_and_stderr(rewards_correlation_values)

    representative_seed_index = 0
    best_seed_index = 0
    valid_seed_indices = [
        summary["seed_index"] for summary in seed_summaries if summary.get("success_metric") is not None
    ]
    if valid_seed_indices:
        best_seed_index = min(
            valid_seed_indices,
            key=lambda index: (
                round(abs(seed_summaries[index]["success_metric"] - success_metric_target), 12),
                index,
            ),
        )
        target_value = success_metric_mean if success_metric_mean is not None else success_metric_target
        representative_seed_index = min(
            valid_seed_indices,
            key=lambda index: (
                round(abs(seed_summaries[index]["success_metric"] - target_value), 12),
                index,
            ),
        )

    feedback_lines: list[str] = []
    for metric_name, metric_sequences in metrics_by_tag.items():
        mean_series, stderr_series = _aggregate_scalar_sequences(metric_sequences)
        if not mean_series:
            continue

        display_name = metric_name.split("Eureka/", 1)[-1]
        if display_name == "success_metric":
            display_name = "task_score"

        sampled_values: list[str] = []
        for index in range(0, len(mean_series), subsampling):
            mean_value = mean_series[index]
            stderr_value = stderr_series[index]
            if len(metric_sequences) > 1:
                sampled_values.append(f"{mean_value:.2f}±{stderr_value:.2f}")
            else:
                sampled_values.append(f"{mean_value:.2f}")

        if has_success_metric and display_name == "oracle_total_rewards":
            continue

        feedback_lines.append(
            f"{display_name}: {sampled_values}, Min: {min(mean_series):.2f}, Max: {max(mean_series):.2f}, Mean:"
            f" {float(sum(mean_series) / len(mean_series)):.2f} \n"
        )

    feedback = "".join(feedback_lines)
    feedback += f"\nThe desired task_score to win is: {success_metric_target:.2f}\n"

    return {
        "feedback": feedback,
        "success_metric_mean": success_metric_mean,
        "success_metric_stderr": success_metric_stderr,
        "rewards_correlation_mean": rewards_correlation_mean,
        "rewards_correlation_stderr": rewards_correlation_stderr,
        "seed_summaries": seed_summaries,
        "best_seed_index": best_seed_index,
        "representative_seed_index": representative_seed_index,
        "num_seeds": len(seed_summaries),
    }


def get_freest_gpu():
    """Get the GPU with the most free memory.

    If a scheduler already constrained device visibility, keep that mapping and let callers use ``cuda`` as-is.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return None
    gpus = GPUtil.getGPUs()
    if not gpus:
        return None
    # Sort GPUs by memory usage
    gpus.sort(key=lambda gpu: gpu.memoryUsed)
    return gpus[0].id


def resolve_sim_device(device: str) -> str:
    """Resolve ``cuda`` to a concrete device when the scheduler has not already pinned visibility."""
    if device != "cuda":
        return device
    device_id = get_freest_gpu()
    return "cuda" if device_id is None else f"cuda:{device_id}"


def bootstrap_observations_via_subprocess(task: str, device: str, env_seed: int = 42) -> str:
    """Fetch ``_get_observations`` source in a dedicated subprocess.

    The bootstrap subprocess writes the source to disk before exiting so the caller is not blocked by Isaac shutdown
    issues in a multiprocessing child.
    """
    with tempfile.TemporaryDirectory(prefix="isaaclab_obs_bootstrap_") as temp_dir:
        output_file = os.path.join(temp_dir, "observations.py")
        cmd = [
            sys.executable,
            "-m",
            "isaaclab_eureka.observation_bootstrap",
            "--task",
            task,
            "--device",
            device,
            "--seed",
            str(env_seed),
            "--output-file",
            output_file,
        ]
        completed = subprocess.run(cmd)
        if os.path.exists(output_file):
            with open(output_file) as file:
                observation_string = file.read()
            if observation_string.strip():
                return observation_string

        raise RuntimeError(
            f"Observation bootstrap failed for task {task!r} with return code {completed.returncode}."
        )


class MuteOutput:
    """Context manager to mute stdout and stderr."""

    def __enter__(self): 
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")  # noqa: SIM115
        sys.stderr = open(os.devnull, "w")  # noqa: SIM115
        return self

    def __exit__(self, *args):
        sys.stdout = self._stdout
        sys.stderr = self._stderr
