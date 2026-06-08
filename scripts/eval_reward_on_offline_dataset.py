# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the TAC score of a *new* candidate reward function against the oracle, using
an offline trajectory dataset previously recorded by ``record_offline_trajectories.py``.

Workflow this script enables:

    1. (one-time, requires Isaac Sim) Run ``record_offline_trajectories.py`` to capture
       per-step robot/task state for a Tacreka training run, every K PPO iterations.
       The recorder also stores the per-step oracle reward, which we *re-use* here.

    2. (offline, plain Python) Run THIS script with any new candidate reward function.
       For each saved snapshot ``eta_t`` (the trajectory distribution induced by the
       policy at PPO iter t), we replay the saved per-step state through the new
       reward function, sum per-episode to get ``E[G_new(tau)]`` estimates, and compute
       the Trajectory Alignment Coefficient between the saved oracle returns and the
       new candidate returns:

           sigma_TAC := tau_b( E[G_oracle], E[G_new] )

       across the snapshots indexing eta_t.

The new candidate reward function is provided either as

  (a) An LLM-style ``def _get_rewards_eureka(self) -> tuple[Tensor, dict]: ...``
      block in a .py file or .json (same convention used elsewhere in this codebase).
      We bind it to a synthetic ``FakeEnv`` whose ``self._robot.data.*`` and
      ``self._desired_pos_w`` attributes mirror the per-step saved tensors exactly
      as the env would have provided them at training time.

  (b) A plain function ``def reward(state: dict[str, Tensor], action: Tensor) -> Tensor``
      that consumes the saved state directly. (See ``--reward_signature plain``.)

Imports inside an LLM-generated reward function (e.g. ``subtract_frame_transforms``,
``quat_from_matrix``) work as long as ``isaaclab.utils.math`` is on the PYTHONPATH and
does not require Isaac Sim at import time -- which is the case for the math utilities
in IsaacLab today.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import types
from typing import Callable

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Load tac_score utilities (compute_tac_from_arrays) without triggering
# isaaclab_eureka/__init__.py (which pulls in Isaac Sim's AppLauncher).
# ---------------------------------------------------------------------------


def _load_tac_score_module():
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.normpath(
        os.path.join(here, "..", "source", "isaaclab_eureka", "isaaclab_eureka", "tac_score.py")
    )
    if not os.path.isfile(candidate):
        raise FileNotFoundError(f"tac_score.py not found at {candidate}")
    spec = importlib.util.spec_from_file_location("_tacreka_tac_score", candidate)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Reward-function loading
# ---------------------------------------------------------------------------


_LLM_REWARD_HEADER = "def _get_rewards_eureka(self)"


def _strip_code_fences(src: str) -> str:
    src = src.strip()
    if src.startswith("```"):
        src = re.sub(r"^```[a-zA-Z]*\n", "", src)
        src = re.sub(r"\n```$", "", src)
    return src


def _read_reward_source(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Reward source file not found: {path}")
    if path.endswith(".json"):
        with open(path) as f:
            data = json.load(f)
        for key in ("reward_function", "gpt_reward_method", "candidate_reward_function"):
            if key in data and isinstance(data[key], str):
                return _strip_code_fences(data[key])
        raise KeyError(f"No reward function field in {path}.")
    with open(path) as f:
        return _strip_code_fences(f.read())


# ---------------------------------------------------------------------------
# FakeEnv: mimics the env attributes an LLM-style reward function reads.
# ---------------------------------------------------------------------------


class _FakeRobotData:
    """Holds tensor attributes named exactly like ``ArticulationData`` fields."""


class _FakeRobot:
    def __init__(self):
        self.data = _FakeRobotData()


class FakeEnv:
    """Minimal stand-in for an IsaacLab ``DirectRLEnv``: just enough to call an
    LLM-generated ``_get_rewards_eureka(self)`` body offline.

    Attribute conventions (matching the Quadcopter env and the recorder's saved schema):
      * ``self._robot.data.<field>`` for fields in the per-step state's robot view, e.g.
        ``root_pos_w``, ``root_quat_w``, ``root_lin_vel_b``, ``root_ang_vel_b``,
        ``projected_gravity_b``.
      * ``self._desired_pos_w`` (and any other ``_*`` keys saved by the recorder) for
        env-level state.
      * ``self.num_envs`` for batch size.
    """

    def __init__(self, num_envs: int):
        self._robot = _FakeRobot()
        self.num_envs = int(num_envs)
        # Saved trajectories are replayed on CPU (``torch.load(..., map_location="cpu")``).
        # Some LLM-generated reward bodies end with ``return total_reward.to(self.device)``,
        # so expose a matching device to keep those calls working offline.
        self.device = torch.device("cpu")

    def set_step_state(self, state_step: dict[str, torch.Tensor]) -> None:
        for name, tensor in state_step.items():
            if name.startswith("_"):
                # env-level (e.g. _desired_pos_w)
                setattr(self, name, tensor)
            else:
                # robot.data.* (e.g. root_pos_w, root_quat_w, ...)
                setattr(self._robot.data, name, tensor)


def _bind_llm_reward(candidate_source: str) -> Callable[[FakeEnv], torch.Tensor]:
    """Compile an LLM-style reward function source and return a callable
    ``compute(fake_env) -> reward[num_envs]``.

    The LLM source typically does ``from isaaclab.utils.math import subtract_frame_transforms``
    or relies on ``from {env_module} import *`` at training time. To match training-time
    semantics as closely as possible without booting Isaac Sim we expose a generous
    namespace: ``torch`` and the contents of ``isaaclab.utils.math`` (which is
    AppLauncher-free).
    """
    import isaaclab.utils.math as iml

    namespace: dict = {"torch": torch}
    for attr in dir(iml):
        if not attr.startswith("_"):
            namespace[attr] = getattr(iml, attr)
    exec(candidate_source, namespace)
    if "_get_rewards_eureka" not in namespace:
        raise ValueError(f"Expected `{_LLM_REWARD_HEADER}` in source; nothing found.")

    fn = namespace["_get_rewards_eureka"]

    def _compute(fake_env: FakeEnv) -> torch.Tensor:
        out = fn(fake_env)
        # LLM convention is ``return total_reward, individual_dict``; tolerate both.
        if isinstance(out, tuple):
            reward = out[0]
        else:
            reward = out
        if not isinstance(reward, torch.Tensor):
            reward = torch.as_tensor(reward)
        return reward

    return _compute


def _load_plain_reward(path: str) -> Callable[[dict, torch.Tensor], torch.Tensor]:
    """Load a plain ``def reward(state: dict, action: Tensor) -> Tensor`` from a .py file."""
    spec = importlib.util.spec_from_file_location("_user_plain_reward", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "reward"):
        raise ValueError(
            f"--reward_signature plain expects a top-level ``reward(state, action)`` in {path}."
        )
    return module.reward


# ---------------------------------------------------------------------------
# Per-snapshot evaluation: recompute new-candidate per-step rewards, aggregate
# into per-episode returns, average across episodes for E[G_new].
# ---------------------------------------------------------------------------


def _aggregate_episode_returns(
    rewards: torch.Tensor,        # [T, num_envs]
    done: torch.Tensor,           # [T, num_envs] bool
    gamma: float,
) -> list[float]:
    """Walk per-env reward streams, slicing on each True in ``done``, returning the list
    of completed-episode (discounted) returns in arrival order.
    """
    T, n_envs = rewards.shape
    completed: list[float] = []
    cur_return = torch.zeros(n_envs, dtype=rewards.dtype)
    cur_discount = torch.ones(n_envs, dtype=rewards.dtype)
    for t in range(T):
        cur_return += cur_discount * rewards[t]
        cur_discount = cur_discount * gamma
        if bool(done[t].any()):
            done_idx = torch.nonzero(done[t], as_tuple=False).flatten()
            for i in done_idx.tolist():
                completed.append(float(cur_return[i].item()))
                cur_return[i] = 0.0
                cur_discount[i] = 1.0
    return completed


def _compute_new_candidate_returns_for_snapshot(
    snap_data: dict,
    new_reward_compute,                    # FakeEnv -> Tensor[num_envs] OR (state,action) -> Tensor
    signature: str,                        # "llm" | "plain"
    gamma: float,
) -> tuple[float, list[float]]:
    """Recompute per-step new-candidate rewards from the saved state and aggregate to
    per-episode returns. Returns (E[G_new], per_episode_returns)."""
    state: dict[str, torch.Tensor] = snap_data["state"]
    action: torch.Tensor = snap_data["action"]
    done: torch.Tensor = snap_data["done"]

    T, n_envs = action.shape[0], action.shape[1]
    rewards = torch.zeros(T, n_envs, dtype=torch.float32)

    if signature == "llm":
        fake_env = FakeEnv(num_envs=n_envs)
        for t in range(T):
            step_state = {name: tensor[t] for name, tensor in state.items()}
            fake_env.set_step_state(step_state)
            r_t = new_reward_compute(fake_env)
            rewards[t] = r_t.detach().to(torch.float32).cpu()
    elif signature == "plain":
        for t in range(T):
            step_state = {name: tensor[t] for name, tensor in state.items()}
            r_t = new_reward_compute(step_state, action[t])
            r_t = torch.as_tensor(r_t).detach().to(torch.float32).cpu()
            rewards[t] = r_t
    else:
        raise ValueError(f"Unknown reward_signature: {signature}")

    per_ep = _aggregate_episode_returns(rewards, done, gamma)
    if not per_ep:
        # Recorder didn't terminate any episodes (e.g. broken-recorder data); fall back
        # to mean per-step reward as the "expected return" estimate.
        return float(rewards.mean().item()), [float(rewards.mean().item())]
    return float(np.mean(per_ep)), per_ep


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _load_dataset_index(dataset_dir: str) -> dict:
    index_path = os.path.join(dataset_dir, "dataset.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"No dataset.json in {dataset_dir}")
    with open(index_path) as f:
        return json.load(f)


def _resolve_snapshot_path(dataset_dir: str, snap: dict) -> str:
    name = snap.get("file") or f"iter_{int(snap['iter']):06d}.pt"
    path = os.path.join(dataset_dir, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Snapshot file missing: {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compute the TAC score of a new candidate reward against the oracle "
        "using a previously-recorded offline trajectory dataset."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory containing dataset.json + iter_*.pt files (one of the per-run output"
        " directories produced by record_offline_trajectories.py).",
    )
    parser.add_argument(
        "--new_reward",
        type=str,
        required=True,
        help="Path to the new candidate reward function (.py or .json).",
    )
    parser.add_argument(
        "--reward_signature",
        type=str,
        default="llm",
        choices=["llm", "plain"],
        help=(
            "'llm' (default): the file defines `_get_rewards_eureka(self)` (LLM-generated style); "
            "'plain': the file defines `reward(state: dict, action: Tensor) -> Tensor`."
        ),
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Discount factor for episode returns. Defaults to dataset.json's gamma.",
    )
    parser.add_argument(
        "--drop_first",
        type=int,
        default=2,
        help="Skip the first N snapshots when computing TAC (early-training noise). "
        "Forwarded to compute_tac_from_arrays. Default: 2.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Where to save the JSON result. Defaults to <dataset_dir>/tac_eval_<reward_basename>.json.",
    )
    parser.add_argument(
        "--use_saved_oracle_summary",
        action="store_true",
        default=True,
        help="Use the per-snapshot expected_oracle_return from dataset.json (recommended). "
        "If unset, recompute from saved per-step oracle_reward + done mask.",
    )
    args = parser.parse_args()

    tac_module = _load_tac_score_module()
    compute_tac_from_arrays = tac_module.compute_tac_from_arrays

    # --- 1. load dataset index ---
    index = _load_dataset_index(args.dataset_dir)
    snapshots = index.get("snapshots", [])
    if not snapshots:
        raise SystemExit(
            f"No snapshots in {args.dataset_dir}/dataset.json. Did the recorder fail?"
        )
    gamma = args.gamma if args.gamma is not None else float(index.get("gamma", 1.0))
    print(f"[EVAL] dataset: {args.dataset_dir}")
    print(f"[EVAL] task   : {index.get('task')}")
    print(f"[EVAL] n_snapshots = {len(snapshots)}, gamma = {gamma}")

    # --- 2. load + bind the new candidate reward function ---
    src = _read_reward_source(args.new_reward)
    if args.reward_signature == "llm":
        if _LLM_REWARD_HEADER not in src:
            raise SystemExit(
                f"--reward_signature llm requires `{_LLM_REWARD_HEADER}` in {args.new_reward}."
            )
        new_reward_compute = _bind_llm_reward(src)
    else:
        new_reward_compute = _load_plain_reward(args.new_reward)
    print(f"[EVAL] new reward loaded ({args.reward_signature}) from {args.new_reward}")

    # --- 3. iterate snapshots ---
    iters: list[int] = []
    oracle_returns: list[float] = []
    new_returns: list[float] = []
    saved_cand_returns: list[float] = []
    per_snap_records: list[dict] = []

    for snap in snapshots:
        path = _resolve_snapshot_path(args.dataset_dir, snap)
        snap_data = torch.load(path, map_location="cpu", weights_only=False)
        it = int(snap_data.get("iter", snap.get("iter")))

        # oracle: use the dataset.json summary unless the user opted out.
        if args.use_saved_oracle_summary and "expected_oracle_return" in snap:
            e_g_oracle = float(snap["expected_oracle_return"])
        else:
            ep_o = _aggregate_episode_returns(snap_data["oracle_reward"], snap_data["done"], gamma)
            e_g_oracle = float(np.mean(ep_o)) if ep_o else float(snap_data["oracle_reward"].mean().item())

        # original (training-time) candidate, for reference
        ep_c_saved = _aggregate_episode_returns(
            snap_data["candidate_reward"], snap_data["done"], gamma
        )
        e_g_cand_saved = (
            float(np.mean(ep_c_saved)) if ep_c_saved else float(snap_data["candidate_reward"].mean().item())
        )

        # new candidate: replay state through the new reward function
        e_g_new, per_ep_new = _compute_new_candidate_returns_for_snapshot(
            snap_data,
            new_reward_compute=new_reward_compute,
            signature=args.reward_signature,
            gamma=gamma,
        )

        iters.append(it)
        oracle_returns.append(e_g_oracle)
        saved_cand_returns.append(e_g_cand_saved)
        new_returns.append(e_g_new)
        per_snap_records.append({
            "iter": it,
            "expected_oracle_return": e_g_oracle,
            "expected_saved_candidate_return": e_g_cand_saved,
            "expected_new_candidate_return": e_g_new,
            "n_episodes_aggregated": int(snap_data.get("n_episodes_completed", 0)),
            "n_env_steps": int(snap_data.get("n_env_steps", 0)),
            "per_episode_new_returns": per_ep_new,
        })
        print(
            f"  iter={it:>5d}  E[G_oracle]={e_g_oracle:+.4f}  "
            f"E[G_cand_saved]={e_g_cand_saved:+.4f}  E[G_cand_new]={e_g_new:+.4f}  "
            f"(n_eps={snap_data.get('n_episodes_completed', 0)}, T={snap_data.get('n_env_steps', 0)})"
        )

    if len(iters) < 3:
        print(
            f"[EVAL][WARN] only {len(iters)} snapshots; TAC needs at least 3 paired points."
            " Output will be NaN."
        )

    # --- 4. compute TAC ---
    tac_new = compute_tac_from_arrays(
        np.asarray(oracle_returns, dtype=np.float64),
        np.asarray(new_returns, dtype=np.float64),
        drop_first=args.drop_first,
    )
    # also report TAC of the original (saved) candidate vs the same oracle, as a sanity check
    tac_saved = compute_tac_from_arrays(
        np.asarray(oracle_returns, dtype=np.float64),
        np.asarray(saved_cand_returns, dtype=np.float64),
        drop_first=args.drop_first,
    )

    def _fmt(d, key):
        v = d.get(key)
        try:
            return f"{float(v):+.6f}"
        except (TypeError, ValueError):
            return "n/a"

    print()
    print("================ TAC results ================")
    print(f"  drop_first = {args.drop_first}")
    print(f"  n_eta used = {tac_new.get('n_eta')}")
    print(f"  NEW candidate vs oracle:")
    print(f"    sigma_TAC (Kendall tau-b)   = {_fmt(tac_new, 'tac_score_kendall_tau_b')}")
    print(f"    Spearman rho                = {_fmt(tac_new, 'spearman_rho')}")
    print(f"    Pearson r                   = {_fmt(tac_new, 'pearson_correlation')}")
    print(f"  Reference (saved candidate vs oracle):")
    print(f"    sigma_TAC (Kendall tau-b)   = {_fmt(tac_saved, 'tac_score_kendall_tau_b')}")
    print(f"    Pearson r                   = {_fmt(tac_saved, 'pearson_correlation')}")
    print("=============================================")

    # --- 5. dump JSON ---
    if args.output is None:
        base = os.path.splitext(os.path.basename(args.new_reward))[0]
        args.output = os.path.join(args.dataset_dir, f"tac_eval_{base}.json")
    out = {
        "dataset_dir": os.path.abspath(args.dataset_dir),
        "task": index.get("task"),
        "gamma": gamma,
        "drop_first": args.drop_first,
        "new_reward_path": os.path.abspath(args.new_reward),
        "reward_signature": args.reward_signature,
        "n_snapshots": len(iters),
        "iters": iters,
        "oracle_returns": oracle_returns,
        "saved_candidate_returns": saved_cand_returns,
        "new_candidate_returns": new_returns,
        "tac_new_vs_oracle": tac_new,
        "tac_saved_vs_oracle": tac_saved,
        "per_snapshot": per_snap_records,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[EVAL] wrote {args.output}")


if __name__ == "__main__":
    main()
