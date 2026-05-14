# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

"""Record offline trajectories every K PPO iterations of a Tacreka training run.

This is the post-training counterpart to ``compute_tac_score.py``: instead of just
ranking trajectory distributions, it *saves* enough per-step state so that any new
candidate reward function can be evaluated offline (without re-training and without
re-rolling-out the env).

For each policy snapshot ``model_<iter>.pt`` (sub-sampled every ``--every`` iterations,
typically 100), this script

  1. Loads the policy.
  2. Rolls out ``--num_episodes`` complete episodes (sharing the env's reset distribution).
  3. Per env-step, records:
       - obs, action, done
       - the oracle reward and the candidate reward (using the patched dual-reward hook,
         identical to the one ``EurekaTaskManager`` installs during training)
       - all robot/task state tensors that any reasonable reward function might use
         (root_pos_w, root_quat_w, root_lin_vel_b, root_ang_vel_b, projected_gravity_b,
         and the task-specific goal ``_desired_pos_w`` for Quadcopter).
  4. Saves one ``iter_<iter>.pt`` file per snapshot, plus a ``dataset.json`` index.

Usage (post-training):

    ./isaaclab.sh -p scripts/record_offline_trajectories.py \
        --best_run_json logs/tacreka_sr/<task>/<run>/best_run.json \
        --every 100 --num_episodes 8 --num_envs 8

Re-evaluating an arbitrary new candidate reward later then becomes a small offline loop:
construct a tensor of per-step new-candidate rewards from the saved state, sum per
episode, average across episodes per snapshot to estimate ``E_{tau ~ eta_t}[G(tau)]``,
then call ``compute_tac_from_arrays(saved_oracle_returns, new_candidate_returns)``.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import re
import types

from isaaclab_eureka.config import TASKS_CFG
from isaaclab_eureka.utils import get_freest_gpu


# ---------------------------------------------------------------------------
# Reward-function loading (same convention as compute_tac_score.py)
# ---------------------------------------------------------------------------

_REWARD_FUNC_HEADER = "def _get_rewards_eureka(self)"


def _load_reward_source(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Reward source file not found: {path}")
    if path.endswith(".json"):
        with open(path) as f:
            data = json.load(f)
        for key in ("reward_function", "gpt_reward_method"):
            if key in data and isinstance(data[key], str):
                src = data[key]
                break
        else:
            raise KeyError(f"No 'reward_function' or 'gpt_reward_method' field in {path}.")
    else:
        with open(path) as f:
            src = f.read()
    src = src.strip()
    if src.startswith("```"):
        src = re.sub(r"^```[a-zA-Z]*\n", "", src)
        src = re.sub(r"\n```$", "", src)
    if _REWARD_FUNC_HEADER not in src:
        raise ValueError(
            f"Candidate reward source must define `{_REWARD_FUNC_HEADER}`. Got:\n{src[:200]}..."
        )
    return src


def _resolve_run_dir_and_reward(args) -> tuple[str, str]:
    """Resolve (run_dir, candidate_reward_source) from CLI args."""
    if args.best_run_json:
        with open(args.best_run_json) as f:
            best = json.load(f)
        run_dir = best.get("training_run_dir") or best.get("training_log_dir")
        if not run_dir or not os.path.isdir(run_dir):
            raise SystemExit(
                f"best_run.json at {args.best_run_json} does not point to a valid training_run_dir."
            )
        candidate_src = _load_reward_source(args.best_run_json)
        return run_dir, candidate_src
    if not args.run_dir or not args.reward_function:
        raise SystemExit(
            "Need either --best_run_json or BOTH --run_dir and --reward_function."
        )
    return args.run_dir, _load_reward_source(args.reward_function)


# ---------------------------------------------------------------------------
# Checkpoint discovery: every K iterations
# ---------------------------------------------------------------------------


def _discover_checkpoints_every(run_dir: str, every: int) -> list[tuple[int, str]]:
    """Find ``model_<iter>.pt`` files where iter is a multiple of ``every`` (plus iter 0)."""
    found: list[tuple[int, str]] = []
    for path in glob.glob(os.path.join(run_dir, "**", "model_*.pt"), recursive=True):
        match = re.search(r"model_(\d+)\.pt$", os.path.basename(path))
        if match:
            it = int(match.group(1))
            if it % every == 0:
                found.append((it, path))
    found.sort(key=lambda t: t[0])
    return found


# ---------------------------------------------------------------------------
# Env patching: dual-reward hook
# ---------------------------------------------------------------------------


def _patch_env_dual_reward(env, candidate_source: str) -> None:
    """Match EurekaTaskManager's TEMPLATE_REWARD_STRING semantics: keep the original task
    reward as the oracle, install the candidate, and store both per-step rewards on
    ``env._tac_step_rewards`` for the rollout loop to read."""
    import torch  # noqa: F401

    base_env = env.unwrapped
    if getattr(base_env, "_tac_patched", False):
        namespace: dict = {}
        exec(
            f"from {base_env.__module__} import *\nimport torch\n" + candidate_source,
            namespace,
        )
        setattr(base_env, "_get_rewards_eureka", types.MethodType(namespace["_get_rewards_eureka"], base_env))
        return

    base_env._get_rewards_oracle = base_env._get_rewards
    namespace: dict = {}
    exec(
        f"from {base_env.__module__} import *\nimport torch\n" + candidate_source,
        namespace,
    )
    setattr(base_env, "_get_rewards_eureka", types.MethodType(namespace["_get_rewards_eureka"], base_env))

    def _hook(self):
        oracle_r = self._get_rewards_oracle()
        cand_r, _ = self._get_rewards_eureka()
        self._tac_step_rewards = (oracle_r.detach(), cand_r.detach())
        return oracle_r

    setattr(base_env, "_get_rewards", types.MethodType(_hook, base_env))
    base_env._tac_patched = True


# ---------------------------------------------------------------------------
# Per-step state extraction
# ---------------------------------------------------------------------------


_ROBOT_DATA_FIELDS = (
    "root_pos_w",
    "root_quat_w",
    "root_lin_vel_w",
    "root_lin_vel_b",
    "root_ang_vel_w",
    "root_ang_vel_b",
    "projected_gravity_b",
)
_ENV_FIELDS = ("_desired_pos_w",)


def _capture_state(base_env) -> dict:
    """Snapshot the tensors most reward functions need. Robust to missing fields."""
    out: dict = {}
    robot_data = getattr(getattr(base_env, "_robot", None), "data", None)
    if robot_data is not None:
        for name in _ROBOT_DATA_FIELDS:
            tensor = getattr(robot_data, name, None)
            if tensor is not None:
                out[name] = tensor.detach().clone().cpu()
    for name in _ENV_FIELDS:
        tensor = getattr(base_env, name, None)
        if tensor is not None and hasattr(tensor, "detach"):
            out[name] = tensor.detach().clone().cpu()
    return out


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------


def _load_rsl_rl_policy(env, task: str, checkpoint: str, device: str):
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from rsl_rl.runners import OnPolicyRunner

    agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = device
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    return runner.get_inference_policy(device=env.unwrapped.device)


# ---------------------------------------------------------------------------
# Main rollout: collect num_episodes complete episodes
# ---------------------------------------------------------------------------


def _extract_obs_tensor(obs):
    """Pull the 'policy' obs tensor out of whatever wrapper format the env returns."""
    if hasattr(obs, "to_dict"):
        return obs["policy"] if "policy" in obs.keys() else next(iter(obs.values()))
    if isinstance(obs, dict):
        return obs.get("policy", next(iter(obs.values())))
    return obs


def _rollout_and_record(
    env,
    policy,
    num_episodes: int,
    gamma: float,
    max_total_steps: int,
):
    """Roll out the policy until ``num_episodes`` complete episodes have terminated.

    Assumes the caller has already triggered a "soft reset" by setting
    ``episode_length_buf[:] = max_episode_length - 1``. We do one **warm-up step here**
    (NOT recorded) so the env's timeout fires across all envs, every env auto-resets to
    a fresh sample of mu, and only after that do we start recording. Without this warm-up
    the very first recorded step would be a 1-step "episode" for every env (this exact
    bug produced the all-1-step trajectories we saw in earlier output).

    Returns a dict of stacked CPU tensors plus per-episode summary returns. Per-step
    tensors are stored in shape ``[T, num_envs, ...]`` where T is the number of env steps
    taken; ``done`` and ``episode_id`` mark which steps belong to which episode.
    """
    import torch

    base_env = env.unwrapped
    n_envs = int(base_env.num_envs)

    obs = env.get_observations()
    obs_tensor = _extract_obs_tensor(obs)

    # ---- Warm-up step: flush the caller's soft-reset so we start recording from mu ----
    with torch.inference_mode():
        try:
            warm_actions = policy(obs)
            warm_step = env.step(warm_actions)
            obs = warm_step[0]
            obs_tensor = _extract_obs_tensor(obs)
        except Exception as warm_err:
            # Non-fatal: fall back to recording from current state.
            print(f"[REC][warn] warm-up step failed ({warm_err!r}); recording from current state.")

    # rolling buffers (per step)
    step_obs: list = []
    step_action: list = []
    step_oracle: list = []
    step_cand: list = []
    step_done: list = []
    step_episode_id: list = []
    step_state: dict[str, list] = {}

    # per-env episode trackers
    cur_episode_id = torch.zeros(n_envs, dtype=torch.long)
    oracle_return_buf = torch.zeros(n_envs)
    cand_return_buf = torch.zeros(n_envs)
    discount_buf = torch.ones(n_envs)

    completed_oracle: list[float] = []
    completed_cand: list[float] = []

    steps = 0
    with torch.inference_mode():
        while len(completed_oracle) < num_episodes and steps < max_total_steps:
            actions = policy(obs)
            step_ret = env.step(actions)
            obs = step_ret[0]
            dones = step_ret[2]
            done_mask = dones.bool() if isinstance(dones, torch.Tensor) else torch.as_tensor(
                dones, dtype=torch.bool
            )
            done_mask_cpu = done_mask.detach().cpu()

            # rewards from the patched env hook
            step_o, step_c = base_env._tac_step_rewards
            step_o_cpu = step_o.detach().cpu()
            step_c_cpu = step_c.detach().cpu()

            # state snapshot AFTER step (i.e. s_{t+1})
            state = _capture_state(base_env)

            # buffers
            step_obs.append(obs_tensor.detach().cpu().clone())
            step_action.append(actions.detach().cpu().clone())
            step_oracle.append(step_o_cpu)
            step_cand.append(step_c_cpu)
            step_done.append(done_mask_cpu)
            step_episode_id.append(cur_episode_id.clone())
            for name, tensor in state.items():
                step_state.setdefault(name, []).append(tensor)

            # accumulate discounted episode returns
            oracle_return_buf += discount_buf * step_o_cpu
            cand_return_buf += discount_buf * step_c_cpu
            discount_buf = discount_buf * gamma

            # finalise episodes that just terminated
            if done_mask_cpu.any():
                done_idx = torch.nonzero(done_mask_cpu, as_tuple=False).flatten()
                for i in done_idx.tolist():
                    completed_oracle.append(float(oracle_return_buf[i].item()))
                    completed_cand.append(float(cand_return_buf[i].item()))
                    if len(completed_oracle) >= num_episodes:
                        break
                # reset per-env trackers for the dones
                oracle_return_buf[done_mask_cpu] = 0.0
                cand_return_buf[done_mask_cpu] = 0.0
                discount_buf[done_mask_cpu] = 1.0
                cur_episode_id[done_mask_cpu] += 1

            obs_tensor = _extract_obs_tensor(obs)
            steps += 1

    if not completed_oracle:
        raise RuntimeError("No episodes completed; raise --num_envs or --max_total_steps.")

    out = {
        "obs": torch.stack(step_obs, dim=0),
        "action": torch.stack(step_action, dim=0),
        "oracle_reward": torch.stack(step_oracle, dim=0),
        "candidate_reward": torch.stack(step_cand, dim=0),
        "done": torch.stack(step_done, dim=0),
        "episode_id": torch.stack(step_episode_id, dim=0),
        "episode_oracle_returns": torch.tensor(completed_oracle),
        "episode_candidate_returns": torch.tensor(completed_cand),
        "n_episodes_completed": int(len(completed_oracle)),
        "n_env_steps": int(steps),
        "gamma": float(gamma),
    }
    if step_state:
        out["state"] = {name: torch.stack(seq, dim=0) for name, seq in step_state.items()}
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Record offline trajectories every K PPO iterations of a Tacreka training run,"
            " saving enough per-step state to evaluate any candidate reward offline."
        )
    )
    parser.add_argument("--task", type=str, default="Isaac-Quadcopter-Direct-v0")
    parser.add_argument(
        "--best_run_json",
        type=str,
        default=None,
        help=(
            "Path to a Tacreka `best_run.json`. Resolves both the rsl_rl run_dir and the"
            " candidate reward function. Mutually exclusive with --run_dir/--reward_function."
        ),
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Directory containing model_*.pt checkpoints (mutually exclusive with --best_run_json).",
    )
    parser.add_argument(
        "--reward_function",
        type=str,
        default=None,
        help="Path to a .py file or .json with the candidate reward function.",
    )
    parser.add_argument("--every", type=int, default=100, help="Snapshot every N PPO iterations.")
    parser.add_argument("--num_episodes", type=int, default=8, help="Episodes per snapshot.")
    parser.add_argument("--num_envs", type=int, default=8, help="Parallel envs during rollout.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor for episode returns.")
    parser.add_argument("--max_total_steps", type=int, default=20_000, help="Safety cap per snapshot.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--env_seed", type=int, default=42)
    parser.add_argument("--rl_library", type=str, default="rsl_rl", choices=["rsl_rl"])
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to save iter_<N>.pt files. Defaults to <run_dir>/offline_trajectories.",
    )
    args = parser.parse_args()

    if args.task not in TASKS_CFG:
        print(f"[WARNING] Task {args.task} not in TASKS_CFG; proceeding anyway.")

    run_dir, candidate_source = _resolve_run_dir_and_reward(args)
    output_dir = args.output_dir or os.path.join(run_dir, "offline_trajectories")
    os.makedirs(output_dir, exist_ok=True)

    checkpoints = _discover_checkpoints_every(run_dir, args.every)
    if not checkpoints:
        all_avail = sorted(
            os.path.basename(p)
            for p in glob.glob(os.path.join(run_dir, "**", "model_*.pt"), recursive=True)
        )[:20]
        print(
            f"[REC][WARN] No checkpoints matching every={args.every} in {run_dir}. "
            f"Available: {all_avail}. Nothing to record; exiting cleanly."
        )
        # write a stub dataset.json so callers can detect the empty state
        os.makedirs(args.output_dir or os.path.join(run_dir, "offline_trajectories"), exist_ok=True)
        stub_path = os.path.join(
            args.output_dir or os.path.join(run_dir, "offline_trajectories"), "dataset.json"
        )
        with open(stub_path, "w") as f:
            json.dump(
                {
                    "task": args.task,
                    "run_dir": run_dir,
                    "every": args.every,
                    "snapshots": [],
                    "note": "no checkpoints matched --every; training may have been shorter than `every` iterations.",
                    "available_checkpoints": all_avail,
                },
                f,
                indent=2,
            )
        return
    print(f"[REC] Found {len(checkpoints)} snapshots at every={args.every} in {run_dir}")
    print(f"[REC] Output: {output_dir}")

    # --- launch Isaac Sim ---
    from isaaclab.app import AppLauncher

    device = args.device
    if device == "cuda":
        gpu_id = get_freest_gpu()
        if gpu_id is not None:
            device = f"cuda:{gpu_id}"
    app_launcher = AppLauncher(headless=True, device=device)
    simulation_app = app_launcher.app

    import gymnasium as gym  # noqa: F401
    import isaaclab_tasks  # noqa: F401
    import torch
    from isaaclab.envs import DirectRLEnvCfg  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(args.task)
    env_cfg.sim.device = device
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.env_seed

    env_unwrapped = gym.make(args.task, cfg=env_cfg)
    _patch_env_dual_reward(env_unwrapped, candidate_source)

    if args.rl_library == "rsl_rl":
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        env = RslRlVecEnvWrapper(env_unwrapped)
    else:
        raise NotImplementedError("Only rsl_rl is supported.")

    base_env = env.unwrapped
    snapshot_records = []

    for it, ckpt_path in checkpoints:
        print(f"[REC] iter={it:>5d}  rolling out {args.num_episodes} eps from {ckpt_path}")

        # force fresh-mu for this snapshot: time out all envs on the next step
        try:
            max_len = int(base_env.max_episode_length)
            base_env.episode_length_buf[:] = max_len - 1
        except Exception:
            pass

        policy = _load_rsl_rl_policy(env, args.task, ckpt_path, device=device)
        traj = _rollout_and_record(
            env=env,
            policy=policy,
            num_episodes=args.num_episodes,
            gamma=args.gamma,
            max_total_steps=args.max_total_steps,
        )
        traj["iter"] = int(it)
        traj["checkpoint_path"] = ckpt_path
        traj["task"] = args.task

        out_path = os.path.join(output_dir, f"iter_{it:06d}.pt")
        torch.save(traj, out_path)
        eo = traj["episode_oracle_returns"]
        ec = traj["episode_candidate_returns"]
        print(
            f"       -> saved {out_path}  ({traj['n_episodes_completed']} eps, "
            f"{traj['n_env_steps']} env-steps)  "
            f"E[G_oracle]={eo.mean().item():+.3f}  E[G_cand]={ec.mean().item():+.3f}"
        )
        snapshot_records.append({
            "iter": int(it),
            "checkpoint_path": ckpt_path,
            "file": os.path.relpath(out_path, output_dir),
            "n_episodes_completed": traj["n_episodes_completed"],
            "n_env_steps": traj["n_env_steps"],
            "expected_oracle_return": float(eo.mean().item()),
            "expected_candidate_return": float(ec.mean().item()),
        })

    # --- write dataset index ---
    index = {
        "task": args.task,
        "run_dir": run_dir,
        "every": args.every,
        "num_episodes_per_snapshot": args.num_episodes,
        "num_envs": args.num_envs,
        "gamma": args.gamma,
        "env_seed": args.env_seed,
        "candidate_reward_function": candidate_source,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "snapshots": snapshot_records,
    }
    index_path = os.path.join(output_dir, "dataset.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"[REC] Wrote {index_path} ({len(snapshot_records)} snapshots)")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
