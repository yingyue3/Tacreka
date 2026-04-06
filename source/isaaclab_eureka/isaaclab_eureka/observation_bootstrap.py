"""Fresh-process bootstrap for fetching a task's observation method source."""

from __future__ import annotations

import argparse
import inspect
import os
import traceback

from isaaclab_eureka.utils import resolve_sim_device


def _write_output_file(output_file: str, contents: str):
    temp_output_file = f"{output_file}.tmp"
    with open(temp_output_file, "w") as file:
        file.write(contents)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp_output_file, output_file)


def main():
    parser = argparse.ArgumentParser(description="Bootstrap Isaac Lab observations in a fresh subprocess.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    try:
        resolved_device = resolve_sim_device(args.device)

        from isaaclab.app import AppLauncher

        app_launcher = AppLauncher(headless=True, device=resolved_device)
        _ = app_launcher.app

        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401
        from isaaclab.envs import DirectRLEnvCfg
        from isaaclab_tasks.utils import parse_env_cfg

        env_cfg: DirectRLEnvCfg = parse_env_cfg(args.task)
        env_cfg.sim.device = resolved_device
        env_cfg.seed = args.seed
        env = gym.make(args.task, cfg=env_cfg)
        observation_string = inspect.getsource(env.unwrapped._get_observations)
        _write_output_file(args.output_file, observation_string)

        # Isaac teardown is the unstable part here. Exit once the payload is durable on disk.
        os._exit(0)
    except Exception:
        print(traceback.format_exc(), flush=True)
        os._exit(1)


if __name__ == "__main__":
    main()
