# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a Cartpole policy and record success rate.

Success is defined by the task config: for Isaac-Cartpole-Direct-v0, an episode
is successful when the episode-length ratio (steps survived / max_episode_length)
is within success_metric_tolerance (0.01) of success_metric_to_win (1.0),
i.e. the pole was balanced for at least 99% of the maximum episode length.

Success rate = (number of successful episodes) / (total number of episodes).

Episode detection strategy
--------------------------
We read `base_env.reset_terminated | base_env.reset_time_outs` directly from the
DirectRLEnv after every step.  This is more reliable than reading the dones from
the RslRlVecEnvWrapper, which may only expose `terminated` (pole-fell) and not
`time_outs` (max-episode-length reached).  For a good policy that never fails, the
wrapper-level `dones` would be all-False, causing the eval loop to never collect any
episodes.

We also save `episode_length_buf` *before* each step because the env auto-resets the
buffer inside `_reset_idx` (called during `step()`), so the value is already gone by
the time `step()` returns.
"""

import argparse

from isaaclab_eureka.config import TASKS_CFG
from isaaclab_eureka.utils import get_freest_gpu


TASK_NAME = "Isaac-Cartpole-Direct-v0"


def _check_and_override_policy_arch(agent_cfg, checkpoint: str):
    """Patch agent_cfg hidden dims to match the checkpoint.

    rsl_rl builds ActorCritic from agent_cfg's hidden dims before loading the
    checkpoint.  If those dims differ from what was used during training the load
    fails.  We read the actual dims from the saved weights and override.
    """
    import torch

    saved = torch.load(checkpoint, map_location="cpu")
    sd = saved.get("model_state_dict", saved)

    actor_weight_shapes = [
        v.shape
        for k, v in sorted(sd.items())
        if k.startswith("actor.") and k.endswith(".weight")
    ]
    if not actor_weight_shapes:
        return agent_cfg

    ckpt_num_obs = actor_weight_shapes[0][1]
    ckpt_num_actions = actor_weight_shapes[-1][0]
    hidden_dims = [shape[0] for shape in actor_weight_shapes[:-1]]

    if hasattr(agent_cfg, "policy"):
        agent_cfg.policy.actor_hidden_dims = hidden_dims
        agent_cfg.policy.critic_hidden_dims = hidden_dims
    else:
        agent_cfg["policy"]["actor_hidden_dims"] = hidden_dims
        agent_cfg["policy"]["critic_hidden_dims"] = hidden_dims

    print(f"[INFO] Checkpoint dims — num_obs={ckpt_num_obs}, num_actions={ckpt_num_actions}, hidden={hidden_dims}")
    return agent_cfg


def eval_success_rate(
    checkpoint: str,
    num_episodes: int = 100,
    num_envs: int = 1,
    device: str = "cuda",
    headless: bool = True,
    rl_library: str = "rsl_rl",
    seed: int = 42,
):
    """Run a Cartpole policy for num_episodes and return success rate.

    Success metric = episode_length / max_episode_length (0..1).
    Success = metric within 0.01 of 1.0, i.e. >= 0.99.

    Returns:
        dict: success_rate, n_success, n_episodes, success_metric_to_win,
              success_metric_tolerance, episode_metrics.
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
    success_metric_to_win = cfg["success_metric_to_win"]        # 1.0
    success_metric_tolerance = cfg["success_metric_tolerance"]  # 0.01

    env_cfg: DirectRLEnvCfg = parse_env_cfg(TASK_NAME)
    env_cfg.sim.device = device
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed

    env = gym.make(TASK_NAME, cfg=env_cfg)

    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    if rl_library == "rsl_rl":
        from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner

        agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(TASK_NAME, "rsl_rl_cfg_entry_point")
        agent_cfg.device = device
        agent_cfg = _check_and_override_policy_arch(agent_cfg, checkpoint)
        env = RslRlVecEnvWrapper(env)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
    else:
        raise NotImplementedError(f"rl_library='{rl_library}' is not supported. Use 'rsl_rl'.")

    base_env = env.unwrapped
    n_success = 0
    episode_metrics = []

    obs = env.get_observations()

    with torch.inference_mode():
        while simulation_app.is_running() and len(episode_metrics) < num_episodes:
            # Save episode_length_buf BEFORE the step.
            # The env auto-resets inside step(), clearing this buffer, so we must
            # capture the length at the point of termination here.
            prev_ep_len = base_env.episode_length_buf.clone()

            actions = policy(obs)
            step_ret = env.step(actions)
            obs = step_ret[0]

            # Read done flags directly from DirectRLEnv.
            # reset_terminated: pole fell (task failure)
            # reset_time_outs: max episode length reached (truncation)
            # This is more reliable than step_ret[2] from the wrapper, which may
            # only include terminated (not time_outs) depending on the rsl_rl version.
            reset_terminated = getattr(base_env, "reset_terminated", None)
            reset_time_outs = getattr(base_env, "reset_time_outs", None)

            if reset_terminated is not None and reset_time_outs is not None:
                done_mask = reset_terminated | reset_time_outs
            else:
                # Fallback: episode_length_buf went back to 0 → env was reset
                done_mask = base_env.episode_length_buf < prev_ep_len

            if not done_mask.any():
                continue

            # One entry per finished environment
            done_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
            if done_ids.dim() == 0:
                done_ids = done_ids.unsqueeze(0)

            for env_idx in done_ids:
                # The env increments episode_length_buf by 1 THEN checks termination,
                # so the final episode length = prev_ep_len + 1, capped at max.
                ep_len = min(float(prev_ep_len[env_idx].item()) + 1.0,
                             float(base_env.max_episode_length))
                metric = ep_len / base_env.max_episode_length
                episode_metrics.append(metric)
                if abs(metric - success_metric_to_win) <= success_metric_tolerance:
                    n_success += 1
                print(f"  Episode {len(episode_metrics):>4}: length_ratio={metric:.4f} "
                      f"({'SUCCESS' if abs(metric - success_metric_to_win) <= success_metric_tolerance else 'FAIL'})")
                if len(episode_metrics) >= num_episodes:
                    break

    env.close()
    simulation_app.close()

    n_episodes = len(episode_metrics)
    success_rate = n_success / n_episodes if n_episodes > 0 else 0.0
    print(f"Success rate: {success_rate}")
    print(f"Number of successful episodes: {n_success}")
    print(f"Number of episodes: {n_episodes}")
    print(f"Success metric to win: {success_metric_to_win}")
    print(f"Success metric tolerance: {success_metric_tolerance}")
    print(f"Episode metrics: {episode_metrics}")

    return {
        "success_rate": success_rate,
        "n_success": n_success,
        "n_episodes": n_episodes,
        "success_metric_to_win": success_metric_to_win,
        "success_metric_tolerance": success_metric_tolerance,
        "episode_metrics": episode_metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a Cartpole policy and record success rate.\n"
            "Success = episode-length ratio >= 0.99 (pole balanced for >=99%% of max episode)."
        )
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to rsl_rl checkpoint (.pt).")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to evaluate.")
    parser.add_argument(
        "--num_envs", type=int, default=1,
        help="Number of parallel envs. Use 1 for strict per-episode success counting.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on.")
    parser.add_argument("--headless", action="store_true", default=True, help="Run without display (default).")
    parser.add_argument("--no_headless", action="store_false", dest="headless", help="Show GUI.")
    parser.add_argument("--rl_library", type=str, default="rsl_rl", choices=["rsl_rl"])
    parser.add_argument("--seed", type=int, default=42, help="Environment seed.")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional path to a text file where results will be written.",
    )
    args = parser.parse_args()

    print(f"Evaluating checkpoint : {args.checkpoint}")
    print(f"Task                  : {TASK_NAME}")
    print(f"Episodes: {args.num_episodes}  |  Envs: {args.num_envs}  |  Seed: {args.seed}")

    result = eval_success_rate(
        checkpoint=args.checkpoint,
        num_episodes=args.num_episodes,
        num_envs=args.num_envs,
        device=args.device,
        headless=args.headless,
        rl_library=args.rl_library,
        seed=args.seed,
    )

    print("\n========== Results ==========")
    print(f"Success rate : {result['success_rate']:.2%}  ({result['n_success']}/{result['n_episodes']} episodes)")
    print(f"Success criterion: episode-length ratio within {result['success_metric_tolerance']} of {result['success_metric_to_win']}")

    if result.get("episode_metrics"):
        import numpy as np
        arr = result["episode_metrics"]
        print(f"Episode-length ratio — min: {min(arr):.4f}, max: {max(arr):.4f}, mean: {float(np.mean(arr)):.4f}")

    if args.output:
        with open(args.output, "w") as f:
            f.write(f"task\t{TASK_NAME}\n")
            f.write(f"checkpoint\t{args.checkpoint}\n")
            f.write(f"success_rate\t{result['success_rate']}\n")
            f.write(f"n_success\t{result['n_success']}\n")
            f.write(f"n_episodes\t{result['n_episodes']}\n")
            f.write(f"success_metric_to_win\t{result['success_metric_to_win']}\n")
            f.write(f"success_metric_tolerance\t{result['success_metric_tolerance']}\n")
            if result.get("episode_metrics"):
                import numpy as np
                arr = result["episode_metrics"]
                f.write(f"episode_ratio_min\t{min(arr):.6f}\n")
                f.write(f"episode_ratio_max\t{max(arr):.6f}\n")
                f.write(f"episode_ratio_mean\t{float(np.mean(arr)):.6f}\n")
        print(f"Wrote results to {args.output}")

    return result


if __name__ == "__main__":
    main()
