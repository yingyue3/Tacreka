# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

"""Trajectory Alignment Coefficient (TAC) computed from Tacreka training logs.

Background
----------
``EurekaTaskManager._prepare_eureka_environment`` patches the env so that on every step
both the original task reward (the *oracle*) and the candidate LLM reward are evaluated
and accumulated into per-env episodic sum buffers (``self._eureka_episode_sums``). On
episode end, the buffers are mean-pooled across the envs that just terminated, divided by
``max_episode_length_s``, and written to TensorBoard as

    Eureka/oracle_total_rewards   ~  E_{tau ~ eta_t}[ G_oracle(tau)   | gamma = 1 ]
    Eureka/eureka_total_rewards   ~  E_{tau ~ eta_t}[ G_candidate(tau)| gamma = 1 ]

at every PPO iteration ``t`` of the training run, where ``eta_t`` is the trajectory
distribution induced by the policy snapshot at iteration ``t`` (all snapshots share the
env's fixed reset distribution ``mu``).

Given this paired sequence ``{(o_t, c_t)}_{t=1..N}`` the Trajectory Alignment Coefficient

    sigma_TAC(D_h, D_{r,gamma}) = (P - Q) / sqrt((P + Q + X0)(P + Q + Y0))

(Kendall's Tau-b between the two rankings of trajectory distributions) reduces to a single
``scipy.stats.kendalltau(o, c, variant="b")`` call. This module provides the helpers to do
that, treating the *oracle* sequence as the ground-truth (``D_h``) and the candidate
sequence as ``D_{r,gamma}``.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np


def _load_tensorboard_scalars(path: str) -> dict:
    """Read scalar (and scalar-tensor) series from a TensorBoard event directory.

    Mirrors ``isaaclab_eureka.learning_curve_utils.load_tensorboard_scalar_series`` but
    is duplicated locally so this module can be imported without triggering the
    ``isaaclab_eureka`` package's heavyweight ``__init__.py`` (which imports the Isaac
    Sim ``AppLauncher``). This keeps the CLI usable from plain ``python``.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from tensorboard.util import tensor_util

    event_acc = EventAccumulator(path)
    event_acc.Reload()

    data: dict[str, list[float]] = defaultdict(list)
    for tag in event_acc.Tags().get("scalars", []):
        for event in event_acc.Scalars(tag):
            data[tag].append(float(event.value))
    for tag in event_acc.Tags().get("tensors", []):
        for event in event_acc.Tensors(tag):
            tensor_value = tensor_util.make_ndarray(event.tensor_proto)
            if getattr(tensor_value, "size", 0) != 1:
                continue
            data[tag].append(float(tensor_value.reshape(-1)[0]))
    return dict(data)


DEFAULT_ORACLE_TAG = "Eureka/oracle_total_rewards"
DEFAULT_CANDIDATE_TAG = "Eureka/eureka_total_rewards"
# Match the convention in tacreka_sr_testing._get_eureka_task_feedback, which drops the
# first two scalar samples because they often contain initialisation outliers.
DEFAULT_DROP_FIRST = 2


def compute_tac_from_arrays(
    oracle_returns: Sequence[float],
    candidate_returns: Sequence[float],
    drop_first: int = DEFAULT_DROP_FIRST,
) -> dict:
    """Compute the Trajectory Alignment Coefficient from two paired return sequences.

    Args:
        oracle_returns: ``[E_{tau ~ eta_t}[G_oracle(tau)] for t = 1..N]``.
        candidate_returns: ``[E_{tau ~ eta_t}[G_candidate(tau)] for t = 1..N]``.
        drop_first: Number of initial samples to discard (these are typically noisy
            initialisation values that do not reflect a meaningfully-trained policy).

    Returns:
        A dictionary with at least ``tac_score_kendall_tau_b`` (the σ_TAC), its p-value,
        Spearman ρ and Pearson r as auxiliary diagnostics, and the number of η used.
    """
    from scipy.stats import kendalltau, spearmanr

    o = np.asarray(oracle_returns, dtype=np.float64).ravel()
    c = np.asarray(candidate_returns, dtype=np.float64).ravel()

    n = min(len(o), len(c))
    o, c = o[:n], c[:n]
    if drop_first > 0:
        o = o[drop_first:]
        c = c[drop_first:]

    valid = np.isfinite(o) & np.isfinite(c)
    o, c = o[valid], c[valid]

    nan_result = {
        "tac_score_kendall_tau_b": float("nan"),
        "tac_p_value": float("nan"),
        "spearman_rho": float("nan"),
        "spearman_p_value": float("nan"),
        "pearson_correlation": float("nan"),
        "n_eta": int(len(o)),
    }
    if len(o) < 2:
        return nan_result

    try:
        tau_b, tau_p = kendalltau(o, c, variant="b")
        rho, rho_p = spearmanr(o, c)
        pearson = float(np.corrcoef(o, c)[0, 1])
    except Exception as exc:  # noqa: BLE001
        nan_result["error"] = str(exc)
        return nan_result

    return {
        "tac_score_kendall_tau_b": float(tau_b) if tau_b is not None else float("nan"),
        "tac_p_value": float(tau_p) if tau_p is not None else float("nan"),
        "spearman_rho": float(rho) if rho is not None else float("nan"),
        "spearman_p_value": float(rho_p) if rho_p is not None else float("nan"),
        "pearson_correlation": pearson,
        "n_eta": int(len(o)),
    }


def _find_tag(data: Mapping[str, list], target_tag: str):
    """Return the first scalar series whose tag *ends with* ``target_tag``.

    rsl_rl/rl_games sometimes prefix scalar tags with the run name, so we use ``endswith``
    matching to stay robust to those prefixes -- this matches the convention in
    ``tacreka_sr_testing._get_eureka_task_feedback``.
    """
    return next((data[key] for key in data if key.endswith(target_tag)), None)


def compute_tac_from_log_dir(
    log_dir: str,
    oracle_tag: str = DEFAULT_ORACLE_TAG,
    candidate_tag: str = DEFAULT_CANDIDATE_TAG,
    drop_first: int = DEFAULT_DROP_FIRST,
) -> dict:
    """Compute σ_TAC for one training run by reading its TensorBoard logs.

    Each TensorBoard step is treated as one trajectory distribution η_t, with the logged
    scalar pair ``(oracle, candidate)`` providing the expected returns under the oracle
    and candidate rewards respectively.
    """
    data = _load_tensorboard_scalars(log_dir)
    oracle_series = _find_tag(data, oracle_tag)
    candidate_series = _find_tag(data, candidate_tag)

    if oracle_series is None or candidate_series is None:
        raise KeyError(
            f"Could not find both '{oracle_tag}' and '{candidate_tag}' in TensorBoard logs"
            f" at {log_dir}. Available tags: {sorted(data.keys())}"
        )

    result = compute_tac_from_arrays(oracle_series, candidate_series, drop_first=drop_first)
    result["log_dir"] = log_dir
    result["oracle_tag"] = oracle_tag
    result["candidate_tag"] = candidate_tag
    return result


def compute_tac_from_best_run_json(
    best_run_json_path: str,
    oracle_tag: str = DEFAULT_ORACLE_TAG,
    candidate_tag: str = DEFAULT_CANDIDATE_TAG,
    drop_first: int = DEFAULT_DROP_FIRST,
) -> dict:
    """Compute σ_TAC from the ``training_log_dir`` referenced in a Tacreka ``best_run.json``."""
    with open(best_run_json_path) as f:
        best = json.load(f)
    log_dir = best.get("training_log_dir") or best.get("log_dir")
    if not log_dir or not os.path.isdir(log_dir):
        raise FileNotFoundError(
            f"`best_run.json` at {best_run_json_path} does not reference a valid training_log_dir."
        )
    result = compute_tac_from_log_dir(
        log_dir,
        oracle_tag=oracle_tag,
        candidate_tag=candidate_tag,
        drop_first=drop_first,
    )
    result["best_run_json"] = best_run_json_path
    result["success_metric"] = best.get("success_metric")
    result["rewards_correlation_pearson_recorded"] = best.get("rewards_correlation")
    return result
