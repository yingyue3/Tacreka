"""Fresh-process seed worker used by task managers.

This module is launched with ``python -m isaaclab_eureka.seed_training_worker`` so each reward/seed pair runs in a
new Python interpreter instead of a forked multiprocessing child.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from contextlib import nullcontext

from isaaclab_eureka.learning_curve_utils import export_learning_curve_artifacts, resolve_checkpoint_path
from isaaclab_eureka.managers.eureka_task_manager import EurekaTaskManager
from isaaclab_eureka.managers.tacreka_task_manager import TacrekaTaskManager
from isaaclab_eureka.utils import MuteOutput


def _write_result_file(result_file: str, result: dict):
    """Persist the worker result before shutting down Isaac."""
    temp_result_file = f"{result_file}.tmp"
    with open(temp_result_file, "w") as file:
        json.dump(result, file, indent=2, default=str)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp_result_file, result_file)


def _build_manager(args: argparse.Namespace):
    if args.manager_type == "eureka":
        manager = EurekaTaskManager.__new__(EurekaTaskManager)
    else:
        manager = TacrekaTaskManager.__new__(TacrekaTaskManager)
        manager._video = args.video
        manager._video_length = args.video_length
        manager._video_interval = args.video_interval

    manager._log_namespace = args.log_namespace
    manager._task = args.task
    manager._rl_library = args.rl_library
    manager._device = args.device
    manager._max_training_iterations = args.max_training_iterations
    manager._env_seed = args.seed
    manager._num_seeds_per_reward = 1
    manager._rl_log_root_dir = os.path.abspath(args.rl_log_root_dir) if args.rl_log_root_dir else None
    manager._idx = args.run_index
    manager._current_env_seed = None
    manager._success_metric_string = args.success_metric_string or ""
    if manager._success_metric_string and not manager._success_metric_string.startswith("extras['Eureka/success_metric']"):
        manager._success_metric_string = "extras['Eureka/success_metric'] = " + manager._success_metric_string
    return manager


def _run_seed(args: argparse.Namespace) -> dict:
    manager = _build_manager(args)
    with open(args.reward_file) as file:
        reward_func_string = file.read()

    try:
        manager._create_environment(args.seed)
        manager._prepare_eureka_environment(reward_func_string)
        context = MuteOutput() if args.run_index > 0 else nullcontext()
        with context:
            manager._run_training(seed=args.seed)

        seed_result = {
            "success": True,
            "seed": args.seed,
            "seed_index": args.seed_index,
            "log_dir": manager._log_dir,
            "run_dir": manager._run_dir,
        }
        checkpoint_file = resolve_checkpoint_path(manager._run_dir)
        if checkpoint_file is not None:
            seed_result["checkpoint_file"] = checkpoint_file
        try:
            learning_curve_dir = os.path.join(manager._run_dir, "learning_curves")
            learning_curve = export_learning_curve_artifacts(
                manager._log_dir,
                output_dir=learning_curve_dir,
                run_name=os.path.basename(manager._run_dir),
            )
            if learning_curve is not None:
                seed_result["learning_curve"] = learning_curve
        except Exception as plot_error:
            seed_result["learning_curve_error"] = str(plot_error)
        _write_result_file(args.result_file, seed_result)
        return seed_result
    except Exception as seed_error:
        print(traceback.format_exc())
        failure_result = {
            "success": False,
            "seed": args.seed,
            "seed_index": args.seed_index,
            "exception": f"Seed {args.seed} failed: {seed_error}",
        }
        _write_result_file(args.result_file, failure_result)
        return failure_result
    finally:
        if hasattr(manager, "_close_environment"):
            manager._close_environment()
        if hasattr(manager, "_simulation_app"):
            manager._simulation_app.close()


def main():
    parser = argparse.ArgumentParser(description="Run one reward/seed training in a fresh Python process.")
    parser.add_argument("--manager-type", choices=["eureka", "tacreka"], required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--rl-library", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--seed-index", type=int, required=True)
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument("--max-training-iterations", type=int, required=True)
    parser.add_argument("--reward-file", required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--success-metric-string", default="")
    parser.add_argument("--rl-log-root-dir", default=None)
    parser.add_argument("--log-namespace", default="eureka")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-length", type=int, default=200)
    parser.add_argument("--video-interval", type=int, default=2000)
    args = parser.parse_args()

    _run_seed(args)


if __name__ == "__main__":
    main()
