# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a Quadcopter policy and record success rate.

Success is defined by the task config: for Isaac-Quadcopter-Direct-v0, an episode
is successful when the mean distance to target is <= success_metric_tolerance (0.2).
Success rate = (number of successful episodes) / (total number of episodes).
"""

import argparse
import types

from isaaclab_eureka.config import TASKS_CFG
from isaaclab_eureka.utils import get_freest_gpu


# Eval-only reset template: records success_metric at episode end without needing
# Eureka reward buffers.
TEMPLATE_RESET_STRING_EVAL = """
from {module_name} import *

@torch.inference_mode()
def _reset_idx(self, env_ids):
    if env_ids is None or len(env_ids) == self.num_envs:
        env_ids = torch.arange(self.num_envs, device=self.device)
    extras = dict()
    {success_metric}
    self._reset_idx_original(env_ids)
    if "log" not in self.extras:
        self.extras["log"] = dict()
    self.extras["log"].update(extras)
"""


TASK_NAME = "Isaac-Quadcopter-Direct-v0"


def _patch_env_for_success_metric(env):
    """Patch the env's _reset_idx to record Eureka/success_metric in extras['log']."""
    import torch

    base_env = env.unwrapped
    success_metric_string = TASKS_CFG[TASK_NAME]["success_metric"]
    success_metric_string = "extras['Eureka/success_metric'] = " + success_metric_string

    base_env._reset_idx_original = base_env._reset_idx
    namespace = {}
    template = TEMPLATE_RESET_STRING_EVAL.format(
        module_name=base_env.__module__,
        success_metric=success_metric_string,
    )
    exec(template, namespace)
    setattr(base_env, "_reset_idx", types.MethodType(namespace["_reset_idx"], base_env))


def eval_success_rate(
    checkpoint: str,
    num_episodes: int = 100,
    num_envs: int = 1,
    device: str = "cuda",
    headless: bool = True,
    rl_library: str = "rsl_rl",
    seed: int = 42,
):
    """Run policy for num_episodes and return success rate.

    For Quadcopter, success = mean distance to target <= success_metric_tolerance (0.2).
    Uses num_envs=1 by default so each episode gives one success_metric; with more envs
    the logged value is the mean over the batch that just finished.

    Returns:
        dict with keys: success_rate (float), n_success (int), n_episodes (int),
        success_metric_tolerance (float), list of per-episode success metrics (optional).
    """
    from isaaclab.app import AppLauncher

    if device == "cuda":
        device = f"cuda:{get_freest_gpu()}"

    app_launcher = AppLauncher(headless=headless, device=device)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import torch
    from isaaclab.envs import DirectRLEnvCfg
    from isaaclab_tasks.utils import parse_env_cfg

    cfg = TASKS_CFG[TASK_NAME]
    success_metric_tolerance = cfg["success_metric_tolerance"]

    env_cfg: DirectRLEnvCfg = parse_env_cfg(TASK_NAME)
    env_cfg.sim.device = device
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed

    env = gym.make(TASK_NAME, cfg=env_cfg)
    _patch_env_for_success_metric(env)

    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    if rl_library == "rsl_rl":
        from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner

        agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(TASK_NAME, "rsl_rl_cfg_entry_point")
        agent_cfg.device = device
        env = RslRlVecEnvWrapper(env)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
    else:
        raise NotImplementedError("Only rsl_rl is supported for this eval script.")

    base_env = env.unwrapped
    n_success = 0
    episode_metrics = []

    obs = env.get_observations()

    with torch.inference_mode():
        while simulation_app.is_running() and len(episode_metrics) < num_episodes:
            actions = policy(obs)
            step_ret = env.step(actions)
            obs = step_ret[0]
            terminated = step_ret[2]
            # Wrapper may return (obs, reward, terminated, info) with no truncated
            truncated = step_ret[3] if len(step_ret) > 3 and not isinstance(step_ret[3], dict) else None

            # Check for finished episodes and record success_metric from env extras
            if isinstance(terminated, torch.Tensor):
                if isinstance(truncated, torch.Tensor):
                    any_done = (terminated | truncated).any().item()
                else:
                    any_done = terminated.any().item()
            else:
                if truncated is not None:
                    any_done = any(terminated) or any(truncated)
                else:
                    any_done = any(terminated)

            if any_done:
                log = getattr(base_env, "extras", {}).get("log", {})
                sm = log.get("Eureka/success_metric")
                if sm is not None:
                    val = sm.item() if hasattr(sm, "item") else float(sm)
                    episode_metrics.append(val)
                    if abs(val - cfg["success_metric_to_win"]) <= success_metric_tolerance:
                        n_success += 1
                    if len(episode_metrics) >= num_episodes:
                        break

    n_episodes = len(episode_metrics)
    if n_episodes == 0:
        success_rate = 0.0
    else:
        success_rate = n_success / n_episodes

    print(f"Success rate: {success_rate:.2%} ({n_success}/{n_episodes} episodes)")
    print(f"Success = mean distance to target <= {success_metric_tolerance}")
    print(f"Success metric to win: {cfg['success_metric_to_win']}")

    env.close()
    simulation_app.close()

    return {
        "success_rate": success_rate,
        "n_success": n_success,
        "n_episodes": n_episodes,
        "success_metric_tolerance": success_metric_tolerance,
        "episode_metrics": episode_metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Quadcopter policy and record success rate (fraction of episodes with distance to target <= tolerance)."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to policy checkpoint (.pt).")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to evaluate.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs (use 1 for strict per-episode success rate).")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--headless", action="store_true", default=True, help="Run without display.")
    parser.add_argument("--no_headless", action="store_false", dest="headless")
    parser.add_argument("--rl_library", type=str, default="rsl_rl", choices=["rsl_rl"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None, help="Optional file path to write success rate and metrics.")
    args = parser.parse_args()

    print(f"Evaluating checkpoint: {args.checkpoint}")

    result = eval_success_rate(
        checkpoint=args.checkpoint,
        num_episodes=args.num_episodes,
        num_envs=args.num_envs,
        device=args.device,
        headless=args.headless,
        rl_library=args.rl_library,
        seed=args.seed,
    )

    print(f"Success rate: {result['success_rate']:.2%} ({result['n_success']}/{result['n_episodes']} episodes)")
    print(f"Success = mean distance to target <= {result['success_metric_tolerance']}")

    if result.get("episode_metrics"):
        import numpy as np
        arr = result["episode_metrics"]
        print(f"Success metric (distance) — min: {min(arr):.4f}, max: {max(arr):.4f}, mean: {np.mean(arr):.4f}")

    if args.output:
        with open(args.output, "w") as f:
            f.write(f"success_rate\t{result['success_rate']}\n")
            f.write(f"n_success\t{result['n_success']}\n")
            f.write(f"n_episodes\t{result['n_episodes']}\n")
            f.write(f"success_metric_tolerance\t{result['success_metric_tolerance']}\n")
        print(f"Wrote results to {args.output}")

    return result


if __name__ == "__main__":
    main()
