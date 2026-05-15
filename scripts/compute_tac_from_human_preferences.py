# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

"""Compute σ_TAC against a *human* preference dataset (Bobu et al. §4.1, partial-ranking case).

Background
----------
``Tacreka_Preference`` records, for every video pair shown to the human:

  * the per-step trajectories that produced each video, saved as ``.pt`` files in
    ``<log_dir>/trajectories/`` using the same schema as
    ``scripts/record_offline_trajectories.py``;
  * the human's strict pairwise preference (winner_index ∈ {0, 1}), appended one
    JSON record per line to ``<log_dir>/human_preferences.jsonl``.

Together those two files materialise υ_h(D_h): an unordered set of trajectory pairs
{η_a, η_b} together with the corresponding human relation η_a ⋄_h η_b ∈ {≻, ≺}.

Given any candidate reward function (r, γ) we then construct

    D_{r,γ}(υ_h, r, γ) = { (η_a ⋄_(r,γ) η_b) | {η_a, η_b} ∈ υ_h(D_h) }

by replaying each saved trajectory through the candidate reward to compute
Ĝ_r(η) = Σ γ^t r(s_t, a_t), forming the candidate-side sign per pair, and counting
P, Q, X_0, Y_0 directly per Eq. (4.1).

Usage
-----
After a Tacreka_Preference run completes::

    python scripts/compute_tac_from_human_preferences.py \\
        --log_dir logs/tacreka_sr/Isaac-Quadcopter-Direct-v0/2026-05-10_22-30-00

The script defaults to scoring the run's own ``best_run.json[gpt_reward_method]``
(i.e. "the method I got"). To audit a different candidate, pass ``--reward_function``::

    python scripts/compute_tac_from_human_preferences.py \\
        --log_dir <log_dir> \\
        --reward_function path/to/other_reward.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from typing import Callable

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Module loaders that bypass isaaclab_eureka/__init__.py (which pulls in
# AppLauncher and forces ``./isaaclab.sh -p`` invocation).
# ---------------------------------------------------------------------------


def _load_path_module(filename: str, mod_name: str):
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.normpath(
        os.path.join(here, "..", "source", "isaaclab_eureka", "isaaclab_eureka", filename)
    )
    if not os.path.isfile(candidate):
        raise FileNotFoundError(f"{filename} not found at {candidate}")
    spec = importlib.util.spec_from_file_location(mod_name, candidate)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_tac = _load_path_module("tac_score.py", "_tacreka_tac_score")
compute_tac_from_pairs = _tac.compute_tac_from_pairs


# ---------------------------------------------------------------------------
# Reward source loading (mirrors eval_reward_on_offline_dataset.py)
# ---------------------------------------------------------------------------


_LLM_REWARD_HEADER = "def _get_rewards_eureka(self)"


def _strip_code_fences(src: str) -> str:
    src = src.strip()
    if src.startswith("```"):
        src = re.sub(r"^```[a-zA-Z]*\n", "", src)
        src = re.sub(r"\n```$", "", src)
    return src


def _read_reward_source(path: str) -> str:
    """Load a reward function source from a .py file or a Tacreka *best_run.json*."""
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
# Offline reward replay (FakeEnv mimics the env attrs an LLM reward reads)
# ---------------------------------------------------------------------------


class _FakeRobotData:
    """Holds tensor attributes named exactly like ArticulationData fields."""


class _FakeRobot:
    def __init__(self):
        self.data = _FakeRobotData()


class FakeEnv:
    """Minimal stand-in for an IsaacLab ``DirectRLEnv`` for offline reward replay."""

    def __init__(self, num_envs: int):
        self._robot = _FakeRobot()
        self.num_envs = int(num_envs)

    def set_step_state(self, state_step: dict[str, torch.Tensor]) -> None:
        for name, tensor in state_step.items():
            if name.startswith("_"):
                setattr(self, name, tensor)
            else:
                setattr(self._robot.data, name, tensor)


def _bind_llm_reward(candidate_source: str) -> Callable[[FakeEnv], torch.Tensor]:
    """Compile an LLM-style ``_get_rewards_eureka`` into a callable.

    Exposes ``torch`` and ``isaaclab.utils.math`` (AppLauncher-free) in the namespace
    so typical LLM-generated bodies (``subtract_frame_transforms``, ``quat_from_matrix``,
    ...) compile and run without booting Isaac Sim.
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
        reward = out[0] if isinstance(out, tuple) else out
        if not isinstance(reward, torch.Tensor):
            reward = torch.as_tensor(reward)
        return reward

    return _compute


# ---------------------------------------------------------------------------
# Per-trajectory aggregation
# ---------------------------------------------------------------------------


def _aggregate_episode_returns(
    rewards: torch.Tensor,        # [T, num_envs]
    done: torch.Tensor,           # [T, num_envs] bool
    gamma: float,
) -> list[float]:
    """Slice per-env reward streams on each True in ``done``, returning the list of
    completed-episode discounted returns.
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


def _expected_return_under_reward(
    snap_data: dict,
    new_reward_compute: Callable[[FakeEnv], torch.Tensor],
    gamma: float,
) -> float:
    """Replay one saved trajectory through the candidate reward and return Ĝ_r(η)
    averaged across whatever complete episodes the trajectory contains.

    Falls back to the mean per-step reward if the trajectory contains no full episode
    (e.g. if recording stopped early). This keeps the pair usable for σ_TAC at the
    cost of a small bias for that pair.
    """
    state: dict[str, torch.Tensor] = snap_data.get("state", {})
    action: torch.Tensor = snap_data["action"]
    done: torch.Tensor = snap_data["done"]
    T, n_envs = action.shape[0], action.shape[1]

    if not state:
        # Without saved state we cannot replay any LLM-style reward.
        raise RuntimeError(
            "Trajectory has no `state` field — recorder did not save robot state. "
            "Re-record with the updated RecordManagerQuad (save_trajectory=True)."
        )

    rewards = torch.zeros(T, n_envs, dtype=torch.float32)
    fake_env = FakeEnv(num_envs=n_envs)
    for t in range(T):
        fake_env.set_step_state({name: tensor[t] for name, tensor in state.items()})
        r_t = new_reward_compute(fake_env)
        rewards[t] = r_t.detach().to(torch.float32).cpu()

    per_ep = _aggregate_episode_returns(rewards, done, gamma)
    if per_ep:
        return float(np.mean(per_ep))
    return float(rewards.mean().item())


# ---------------------------------------------------------------------------
# Preference dataset I/O
# ---------------------------------------------------------------------------


def _load_jsonl(path: str) -> list[dict]:
    out: list[dict] = []
    with open(path) as f:
        for ln, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                print(f"[WARN] {path}:{ln} skipped (invalid JSON): {exc}", file=sys.stderr)
    return out


def _resolve_trajectory_path(log_dir: str, traj_path: str | None) -> str | None:
    if not traj_path:
        return None
    if os.path.isabs(traj_path) and os.path.isfile(traj_path):
        return traj_path
    candidate = os.path.join(log_dir, traj_path)
    if os.path.isfile(candidate):
        return candidate
    if os.path.isfile(traj_path):
        return traj_path
    return None


def _resolve_reward_source(args, log_dir: str) -> tuple[str, str]:
    """Return (reward_source_text, source_label_for_logging)."""
    if args.reward_function:
        return _read_reward_source(args.reward_function), os.path.abspath(args.reward_function)
    best_run_json = args.best_run_json or os.path.join(log_dir, "best_run.json")
    if not os.path.isfile(best_run_json):
        raise SystemExit(
            f"No --reward_function and no best_run.json at {best_run_json}. "
            "Either pass --reward_function or point --log_dir at a finished run."
        )
    return _read_reward_source(best_run_json), os.path.abspath(best_run_json)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute σ_TAC (Bobu et al. §4.1, partial-ranking case) between a candidate "
            "reward function and the human preference dataset recorded by Tacreka_Preference."
        )
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Tacreka_Preference run log dir, containing `human_preferences.jsonl` and "
        "`trajectories/`. Defaults the reward to <log_dir>/best_run.json.",
    )
    parser.add_argument(
        "--preferences",
        type=str,
        default=None,
        help="Override path to the JSONL preference file (default: <log_dir>/human_preferences.jsonl).",
    )
    parser.add_argument(
        "--reward_function",
        type=str,
        default=None,
        help="Override path to the candidate reward (.py or .json). "
        "If omitted, uses <log_dir>/best_run.json[gpt_reward_method].",
    )
    parser.add_argument(
        "--best_run_json",
        type=str,
        default=None,
        help="Explicit best_run.json path (overrides the <log_dir> default).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Discount factor used to recompute Ĝ_r(η). "
        "Defaults to each saved trajectory's recorded gamma.",
    )
    parser.add_argument(
        "--indifference_eps",
        type=float,
        default=0.0,
        help="If |Ĝ_r(η_a) − Ĝ_r(η_b)| ≤ eps, treat the candidate side as ∼ "
        "(contributes to X0 instead of P/Q). Default: 0 (strict).",
    )
    parser.add_argument(
        "--include_context",
        type=str,
        nargs="+",
        default=None,
        choices=["iteration_internal", "best_run_comparison"],
        help="Optional filter: only score preference pairs from these contexts.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON report path. Defaults to <log_dir>/tac_human_<reward_basename>.json.",
    )
    args = parser.parse_args()

    log_dir = os.path.abspath(args.log_dir)
    if not os.path.isdir(log_dir):
        raise SystemExit(f"--log_dir not a directory: {log_dir}")

    pref_path = args.preferences or os.path.join(log_dir, "human_preferences.jsonl")
    if not os.path.isfile(pref_path):
        raise SystemExit(f"No preference file at {pref_path}")

    records = _load_jsonl(pref_path)
    if not records:
        raise SystemExit(f"No preference records in {pref_path}")

    if args.include_context:
        records = [r for r in records if r.get("context") in args.include_context]
        if not records:
            raise SystemExit(
                f"No records left after filtering for contexts={args.include_context}."
            )

    reward_src, reward_label = _resolve_reward_source(args, log_dir)
    print(f"[TAC-H] log_dir         : {log_dir}")
    print(f"[TAC-H] preferences     : {pref_path}")
    print(f"[TAC-H] reward source   : {reward_label}")
    print(f"[TAC-H] n records       : {len(records)}")

    new_reward_compute = _bind_llm_reward(reward_src)

    # Per-trajectory cache: replaying the same .pt twice is the common case
    # (the same trajectory often appears in multiple pairs).
    return_cache: dict[tuple[str, float], float] = {}

    def _expected_return(traj_path: str, gamma: float) -> float:
        key = (traj_path, gamma)
        if key in return_cache:
            return return_cache[key]
        snap = torch.load(traj_path, map_location="cpu", weights_only=False)
        g = float(args.gamma) if args.gamma is not None else float(snap.get("gamma", 1.0))
        val = _expected_return_under_reward(snap, new_reward_compute, gamma=g)
        return_cache[key] = val
        return val

    pair_records: list[dict] = []
    skipped: list[dict] = []
    human_signs: list[float] = []
    cand_diffs: list[float] = []

    for rec in records:
        traj_a = _resolve_trajectory_path(log_dir, rec.get("trajectory_a"))
        traj_b = _resolve_trajectory_path(log_dir, rec.get("trajectory_b"))
        if traj_a is None or traj_b is None or not os.path.isfile(traj_a) or not os.path.isfile(traj_b):
            skipped.append({**rec, "skip_reason": "missing_trajectory_file"})
            continue

        winner = rec.get("winner_index")
        if winner not in (0, 1):
            skipped.append({**rec, "skip_reason": f"unsupported_winner_index={winner}"})
            continue

        # Per-trajectory gamma is read from the .pt; --gamma overrides if given.
        snap_a = torch.load(traj_a, map_location="cpu", weights_only=False)
        gamma = float(args.gamma) if args.gamma is not None else float(snap_a.get("gamma", 1.0))
        e_g_a = _expected_return(traj_a, gamma)
        e_g_b = _expected_return(traj_b, gamma)

        # human sign: +1 if A preferred, -1 if B preferred.
        h = +1.0 if winner == 0 else -1.0
        diff = e_g_a - e_g_b

        human_signs.append(h)
        cand_diffs.append(diff)
        pair_records.append({
            "iter": rec.get("iter"),
            "context": rec.get("context"),
            "trajectory_a": traj_a,
            "trajectory_b": traj_b,
            "label_a": rec.get("label_a"),
            "label_b": rec.get("label_b"),
            "winner_index": int(winner),
            "human_sign": h,
            "expected_return_a": e_g_a,
            "expected_return_b": e_g_b,
            "candidate_diff_a_minus_b": diff,
            "gamma": gamma,
        })

    if not pair_records:
        raise SystemExit("No usable preference pairs (all skipped).")

    tac = compute_tac_from_pairs(
        human_signs=np.asarray(human_signs, dtype=np.float64),
        candidate_diffs=np.asarray(cand_diffs, dtype=np.float64),
        indifference_eps=args.indifference_eps,
    )

    # Pretty-print summary.
    n_used = tac["n_pairs_used"]
    print()
    print("=" * 72)
    print(f"  Pairs total / used        : {tac['n_pairs_total']} / {n_used}")
    print(f"  Indifference eps          : {tac['indifference_eps']}")
    print(f"  P (concordant)            : {tac['P']}")
    print(f"  Q (discordant)            : {tac['Q']}")
    print(f"  X0 (cand tied, human not) : {tac['X0']}")
    print(f"  Y0 (human tied, cand not) : {tac['Y0']}")
    tau_str = (
        f"{tac['tac_score_kendall_tau_b']:+.4f}"
        if tac["tac_score_kendall_tau_b"] == tac["tac_score_kendall_tau_b"]  # not NaN
        else "nan"
    )
    print(f"  σ_TAC (Kendall tau-b)     : {tau_str}")
    print("=" * 72)
    if skipped:
        print(f"  ({len(skipped)} preference records skipped — see report for reasons)")

    # Build report
    if args.output is None:
        if args.reward_function:
            base = os.path.splitext(os.path.basename(args.reward_function))[0]
        else:
            base = "best_run"
        args.output = os.path.join(log_dir, f"tac_human_{base}.json")
    report = {
        "log_dir": log_dir,
        "preferences_file": os.path.abspath(pref_path),
        "reward_function_source_path": reward_label,
        "n_records_input": len(records),
        "n_pairs_used": n_used,
        "n_pairs_skipped": len(skipped),
        "indifference_eps": float(args.indifference_eps),
        "include_context": args.include_context,
        "tac_result": tac,
        "pair_records": pair_records,
        "skipped_records": skipped,
    }
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[TAC-H] wrote report: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
