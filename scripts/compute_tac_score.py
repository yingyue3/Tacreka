# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

"""Compute the Trajectory Alignment Coefficient (TAC) of a candidate reward function.

The TAC, introduced in Bobu et al., is a Kendall's Tau-b correlation between two rankings
over a set of trajectory distributions {eta_1, ..., eta_N} that share the same initial-state
distribution mu:

    eta_A  preceq_(r,gamma)  eta_B   <=>   E_{tau ~ eta_A}[G_r(tau)] >= E_{tau ~ eta_B}[G_r(tau)]

In the standard formulation the ground-truth ranking comes from a human; here we substitute the
human ranking with the *oracle* ranking induced by the task's original (hand-engineered) reward
function -- i.e. the one defined in the IsaacLab task. The oracle therefore plays the role of
``D_h`` in the paper, and the candidate (e.g. an LLM-generated) reward provides ``D_{r,gamma}``.

How a trajectory distribution is materialised
---------------------------------------------
Each policy checkpoint defines a (deterministic-by-action / stochastic-by-environment)
trajectory distribution. We obtain trajectories by rolling out that policy on the task with a
fixed env seed (so all policies share the same mu), then estimate the expected return
``E_tau[G_r(tau)]`` for both the oracle and the candidate reward.

You can also include the random-action policy as an extra eta to broaden the spread of
trajectory distributions used to compute the rank correlation.

Usage
-----
    ./isaaclab.sh -p scripts/compute_tac_score.py \
        --task Isaac-Quadcopter-Direct-v0 \
        --reward_function logs/tacreka_sr/<task>/<run>/best_run.json \
        --checkpoint_dir logs/tacreka_sr/<task>/<run>/rl_runs \
        --num_episodes 50 --num_envs 16 --gamma 0.99 \
        --include_random_policy \
        --output tac_results.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import types

# we import this here to avoid GLIBCXX errors when launching Isaac Sim
from isaaclab_eureka.config import TASKS_CFG
from isaaclab_eureka.utils import get_freest_gpu


# ---------------------------------------------------------------------------
# Reward-function loading
# ---------------------------------------------------------------------------

_REWARD_FUNC_HEADER = "def _get_rewards_eureka(self)"


def _load_reward_source(path: str) -> str:
    """Read a candidate reward function source from a .py or a *best_run.json* file.

    Returns the source string of a function whose first line is
    ``def _get_rewards_eureka(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:``
    """
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
            raise KeyError(
                f"No 'reward_function' or 'gpt_reward_method' field in {path}."
            )
    else:
        with open(path) as f:
            src = f.read()

    src = src.strip()
    # strip ```python fences if user pasted a markdown block
    if src.startswith("```"):
        src = re.sub(r"^```[a-zA-Z]*\n", "", src)
        src = re.sub(r"\n```$", "", src)
    if _REWARD_FUNC_HEADER not in src:
        raise ValueError(
            "Candidate reward source must define"
            f" `{_REWARD_FUNC_HEADER}`. Got:\n{src[:200]}..."
        )
    return src


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------


def _discover_checkpoints(checkpoint_dir: str, every: int = 1) -> list[str]:
    """Recursively find ``model_<n>.pt`` files and sort them by training step.

    With ``every>1`` we sub-sample to reduce the number of trajectory distributions used
    (e.g. ``every=10`` keeps every 10th checkpoint per training run, sorted by step).
    """
    checkpoints: list[tuple[int, str]] = []
    for path in glob.glob(os.path.join(checkpoint_dir, "**", "model_*.pt"), recursive=True):
        match = re.search(r"model_(\d+)\.pt$", os.path.basename(path))
        if match:
            checkpoints.append((int(match.group(1)), path))
    checkpoints.sort(key=lambda t: (os.path.dirname(t[1]), t[0]))
    sampled = [p for i, (_, p) in enumerate(checkpoints) if i % every == 0]
    return sampled


# ---------------------------------------------------------------------------
# Env patching: dual-reward hook
# ---------------------------------------------------------------------------


def _patch_env_dual_reward(env, candidate_source: str) -> None:
    """Replace ``env._get_rewards`` with a hook that computes both oracle and candidate rewards.

    The original reward is preserved as ``env._get_rewards_oracle``; the candidate is exec'd as
    ``env._get_rewards_eureka``. After every env step, the per-env tensors of oracle and
    candidate rewards for the *most recent step* are stored on ``env._tac_step_rewards``.
    """
    import torch  # noqa: F401  (kept for parity with task manager template)

    base_env = env.unwrapped

    if hasattr(base_env, "_tac_patched") and base_env._tac_patched:
        # already patched -- just swap the candidate reward function
        namespace: dict = {}
        src = f"from {base_env.__module__} import * \nimport torch\n" + candidate_source
        exec(src, namespace)
        setattr(base_env, "_get_rewards_eureka", types.MethodType(namespace["_get_rewards_eureka"], base_env))
        return

    # 1) preserve the original reward function as the oracle
    base_env._get_rewards_oracle = base_env._get_rewards

    # 2) install the candidate reward function
    namespace: dict = {}
    src = f"from {base_env.__module__} import * \nimport torch\n" + candidate_source
    exec(src, namespace)
    setattr(base_env, "_get_rewards_eureka", types.MethodType(namespace["_get_rewards_eureka"], base_env))

    # 3) install the dual hook that records both rewards each step
    def _get_rewards_hook(self):
        oracle_r = self._get_rewards_oracle()
        cand_r, _ = self._get_rewards_eureka()
        # detach to avoid keeping graph state alive across rollouts
        self._tac_step_rewards = (oracle_r.detach(), cand_r.detach())
        # the policy is fixed (inference only) so the returned reward does not affect dynamics
        return oracle_r

    setattr(base_env, "_get_rewards", types.MethodType(_get_rewards_hook, base_env))
    base_env._tac_patched = True


# ---------------------------------------------------------------------------
# Rollout helpers
# ---------------------------------------------------------------------------


def _make_random_policy(base_env, action_space, device, default_range: float = 1.0):
    """Return a callable that produces uniform-random actions matching the env action space.

    IsaacLab direct envs commonly expose ``Box(-inf, +inf)`` action spaces (the env clamps
    internally), so we fall back to ``[-default_range, +default_range]`` whenever the
    bounds are non-finite. The number of envs is always read directly from
    ``base_env.num_envs`` to avoid any TensorDict / dict-shape ambiguity.
    """
    import math

    import numpy as np
    import torch

    raw_low = np.asarray(action_space.low, dtype=np.float32)
    raw_high = np.asarray(action_space.high, dtype=np.float32)
    if not np.all(np.isfinite(raw_low)) or not np.all(np.isfinite(raw_high)):
        low, high = -default_range, default_range
    else:
        low = float(raw_low.min())
        high = float(raw_high.max())
        if not (math.isfinite(low) and math.isfinite(high)) or low >= high:
            low, high = -default_range, default_range

    # action_space.shape is typically (num_envs, action_dim); fall back gracefully
    if action_space.shape and len(action_space.shape) >= 2:
        action_dim = int(action_space.shape[-1])
    elif action_space.shape:
        action_dim = int(action_space.shape[0])
    else:
        action_dim = int(getattr(base_env, "num_actions", 1))

    n_envs = int(base_env.num_envs)

    def policy(obs):
        return torch.empty(n_envs, action_dim, device=device).uniform_(low, high)

    return policy


def _load_rsl_rl_policy(env, task: str, checkpoint: str, device: str):
    """Load an rsl_rl OnPolicyRunner inference policy from a checkpoint."""
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from rsl_rl.runners import OnPolicyRunner

    agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = device

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    return runner.get_inference_policy(device=env.unwrapped.device)


def _rollout_expected_returns(
    env,
    policy,
    num_episodes: int,
    gamma: float,
    seed: int,
    max_total_steps: int = 200_000,
) -> tuple[float, float, int]:
    """Roll out ``policy`` on ``env`` and return (E[G_oracle], E[G_candidate], n_episodes).
    Episodes are accumulated across the parallel envs until ``num_episodes`` complete ones
    have been seen (or ``max_total_steps`` is reached as a safety cap).
    """
    import torch

    base_env = env.unwrapped
    device = base_env.device
    n_envs = base_env.num_envs

    # Episodes auto-reset inside env.step when done, so the shared initial-state
    # distribution mu is sampled by the env's own reset logic. We do not call
    # env.reset() explicitly between policies because that interacts badly with
    # the wrapped vec-env state on Isaac Sim. To reduce starting-state bias from
    # the previous policy we instead force a full reset by setting episode_length_buf
    # to max_episode_length so all envs time out on the next step.
    try:
        max_len = int(base_env.max_episode_length)
        base_env.episode_length_buf[:] = max_len - 1
    except Exception:
        pass
    obs = env.get_observations()

    oracle_returns_buf = torch.zeros(n_envs, device=device)
    cand_returns_buf = torch.zeros(n_envs, device=device)
    discount_buf = torch.ones(n_envs, device=device)

    completed_oracle: list[float] = []
    completed_cand: list[float] = []

    steps = 0
    with torch.inference_mode():
        while len(completed_oracle) < num_episodes and steps < max_total_steps:
            actions = policy(obs)
            step_ret = env.step(actions)
            obs = step_ret[0]
            dones = step_ret[2]
            if isinstance(dones, torch.Tensor):
                done_mask = dones.bool()
            else:
                done_mask = torch.as_tensor(dones, device=device, dtype=torch.bool)

            step_oracle, step_cand = base_env._tac_step_rewards
            oracle_returns_buf += discount_buf * step_oracle
            cand_returns_buf += discount_buf * step_cand
            discount_buf = discount_buf * gamma

            if done_mask.any():
                done_idx = torch.nonzero(done_mask, as_tuple=False).flatten()
                for i in done_idx.tolist():
                    completed_oracle.append(float(oracle_returns_buf[i].item()))
                    completed_cand.append(float(cand_returns_buf[i].item()))
                    if len(completed_oracle) >= num_episodes:
                        break
                oracle_returns_buf[done_mask] = 0.0
                cand_returns_buf[done_mask] = 0.0
                discount_buf[done_mask] = 1.0

            steps += 1

    if not completed_oracle:
        raise RuntimeError(
            "No episodes completed during rollout -- increase max_total_steps or num_envs."
        )
    n = len(completed_oracle)
    return (
        sum(completed_oracle) / n,
        sum(completed_cand) / n,
        n,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute the Trajectory Alignment Coefficient (Kendall's Tau-b) between a candidate"
            " reward function and the task's oracle reward, over a set of policy-induced"
            " trajectory distributions."
        )
    )
    parser.add_argument("--task", type=str, default="Isaac-Quadcopter-Direct-v0")
    parser.add_argument(
        "--reward_function",
        type=str,
        required=True,
        help=(
            "Path to the candidate reward function. Either a .py file containing"
            " `def _get_rewards_eureka(self) -> ...` or a `best_run.json` with a"
            " `reward_function` / `gpt_reward_method` field."
        ),
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Explicit list of policy checkpoint files (.pt). Each defines one trajectory"
            " distribution eta_k. Mutually exclusive with --checkpoint_dir."
        ),
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory to recursively scan for `model_*.pt` checkpoints.",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=1,
        help="Subsample factor when discovering checkpoints from a directory (1 = keep all).",
    )
    parser.add_argument(
        "--include_random_policy",
        action="store_true",
        help="Add a uniform-random action policy as an extra trajectory distribution.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=64,
        help="Number of complete episodes per policy used to estimate E_tau[G_r(tau)].",
    )
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for G_r(tau).")
    parser.add_argument(
        "--max_total_steps",
        type=int,
        default=200_000,
        help="Safety cap on env steps per policy rollout.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--env_seed", type=int, default=42, help="Shared initial-state-distribution seed.")
    parser.add_argument("--rl_library", type=str, default="rsl_rl", choices=["rsl_rl"])
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    if args.task not in TASKS_CFG:
        print(f"[WARNING] Task {args.task} not in TASKS_CFG; proceeding anyway.")

    if args.checkpoints and args.checkpoint_dir:
        raise SystemExit("Pass either --checkpoints or --checkpoint_dir, not both.")

    if args.checkpoints:
        checkpoints = list(args.checkpoints)
    elif args.checkpoint_dir:
        checkpoints = _discover_checkpoints(args.checkpoint_dir, every=args.checkpoint_every)
    else:
        checkpoints = []

    if not checkpoints and not args.include_random_policy:
        raise SystemExit(
            "Need at least one trajectory distribution: pass --checkpoints, --checkpoint_dir,"
            " and/or --include_random_policy."
        )

    # build the list of (label, kind, payload) trajectory-distribution descriptors
    trajectory_distributions: list[tuple[str, str, str | None]] = []
    if args.include_random_policy:
        trajectory_distributions.append(("random", "random", None))
    for ckpt in checkpoints:
        label = os.path.relpath(ckpt)
        trajectory_distributions.append((label, "checkpoint", ckpt))

    if len(trajectory_distributions) < 2:
        raise SystemExit(
            "Need at least 2 trajectory distributions to compute a rank correlation."
            f" Got {len(trajectory_distributions)}."
        )

    candidate_source = _load_reward_source(args.reward_function)

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
    import numpy as np
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
        raise NotImplementedError("Only rsl_rl is supported for TAC computation.")

    random_policy = _make_random_policy(
        env.unwrapped, env_unwrapped.action_space, device=env.unwrapped.device
    )

    oracle_means: list[float] = []
    candidate_means: list[float] = []
    per_eta_records: list[dict] = []

    for label, kind, payload in trajectory_distributions:
        print(f"[TAC] Rolling out eta = {label} ({kind})")
        if kind == "random":
            policy = random_policy
        else:
            policy = _load_rsl_rl_policy(env, args.task, payload, device=device)

        oracle_mean, cand_mean, n_eps = _rollout_expected_returns(
            env=env,
            policy=policy,
            num_episodes=args.num_episodes,
            gamma=args.gamma,
            seed=args.env_seed,
            max_total_steps=args.max_total_steps,
        )
        print(
            f"       -> n_episodes={n_eps}  E[G_oracle]={oracle_mean:+.4f}"
            f"  E[G_candidate]={cand_mean:+.4f}"
        )
        oracle_means.append(oracle_mean)
        candidate_means.append(cand_mean)
        per_eta_records.append({
            "label": label,
            "kind": kind,
            "checkpoint": payload,
            "n_episodes": n_eps,
            "expected_oracle_return": oracle_mean,
            "expected_candidate_return": cand_mean,
        })

    # --- compute Kendall's Tau-b (the TAC) ---
    from scipy.stats import kendalltau, spearmanr

    oracle_arr = np.asarray(oracle_means, dtype=np.float64)
    cand_arr = np.asarray(candidate_means, dtype=np.float64)
    tau_b, tau_p = kendalltau(oracle_arr, cand_arr, variant="b")
    rho, rho_p = spearmanr(oracle_arr, cand_arr)
    pearson = float(np.corrcoef(oracle_arr, cand_arr)[0, 1]) if len(oracle_arr) > 1 else float("nan")

    # rank summary -- sort eta from lowest oracle return to highest, show both rankings
    oracle_rank = np.argsort(np.argsort(oracle_arr))
    candidate_rank = np.argsort(np.argsort(cand_arr))
    for rec, orank, crank in zip(per_eta_records, oracle_rank.tolist(), candidate_rank.tolist()):
        rec["oracle_rank"] = int(orank)
        rec["candidate_rank"] = int(crank)

    result = {
        "task": args.task,
        "reward_function_path": os.path.abspath(args.reward_function),
        "num_trajectory_distributions": len(trajectory_distributions),
        "num_episodes_per_eta": args.num_episodes,
        "num_envs": args.num_envs,
        "gamma": args.gamma,
        "env_seed": args.env_seed,
        "tac_score_kendall_tau_b": float(tau_b) if tau_b is not None else None,
        "tac_score_p_value": float(tau_p) if tau_p is not None else None,
        "spearman_rho": float(rho) if rho is not None else None,
        "spearman_p_value": float(rho_p) if rho_p is not None else None,
        "pearson_correlation": pearson,
        "trajectory_distributions": per_eta_records,
    }

    print()
    print("=" * 72)
    print(f"TAC score (Kendall's Tau-b)        : {result['tac_score_kendall_tau_b']:+.4f}")
    print(f"  p-value                          : {result['tac_score_p_value']}")
    print(f"Spearman rho (rank corr, secondary): {result['spearman_rho']:+.4f}")
    print(f"Pearson r    (return corr, sanity) : {result['pearson_correlation']:+.4f}")
    print("=" * 72)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote TAC results to {args.output}")

    env.close()
    simulation_app.close()
    return result


if __name__ == "__main__":
    main()
