# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

"""Compute the Trajectory Alignment Coefficient (TAC) from already-logged Tacreka runs.

Background -- and why this script exists separately from ``compute_tac_score.py``:

    ``EurekaTaskManager._prepare_eureka_environment`` patches the env so that on every
    step both the oracle (original task) reward and the candidate (LLM) reward are
    accumulated into per-env episode-sum buffers and emitted to TensorBoard at every PPO
    iteration as

        Eureka/oracle_total_rewards   ~  E_{tau ~ eta_t}[ G_oracle(tau) ]
        Eureka/eureka_total_rewards   ~  E_{tau ~ eta_t}[ G_candidate(tau) ]

    where eta_t is the trajectory distribution induced by the policy at PPO iteration t.
    Because the env's reset distribution mu is fixed across t, all eta_t share the same
    initial-state distribution -- exactly the setting in which Bobu et al. define the
    Trajectory Alignment Coefficient. The candidate-vs-oracle ranking comparison reduces
    to one ``kendalltau(o, c, variant="b")`` call.

This script is therefore *fast* (no Isaac Sim launch) and operates directly on existing
log directories or ``best_run.json`` files produced by ``Tacreka_SR``.

Examples
--------
Single training run:

    python scripts/compute_tac_from_logs.py \
        --input logs/tacreka_sr/Isaac-Quadcopter-Direct-v0/2026-05-03_20-17-53/best_run.json

Multiple inputs (a mix of best_run.json files and raw rl_run dirs):

    python scripts/compute_tac_from_logs.py \
        --input logs/tacreka_sr/.../best_run.json \
        --input logs/tacreka_sr/.../rl_runs/<some_run_dir> \
        --output tac_results.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys


def _load_tac_score_module():
    """Load ``tac_score.py`` directly from the package source.

    We bypass ``isaaclab_eureka/__init__.py`` because that pulls in the Isaac Sim
    ``AppLauncher``, which would force this otherwise-pure-numpy script to be invoked
    via ``./isaaclab.sh -p``. Loading the file by path keeps the script runnable with
    plain ``python``.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.normpath(
        os.path.join(here, "..", "source", "isaaclab_eureka", "isaaclab_eureka", "tac_score.py")
    )
    if not os.path.isfile(candidate):
        # Fallback to a regular import (works when launched via isaaclab.sh).
        from isaaclab_eureka.tac_score import (  # type: ignore  # noqa: F401
            DEFAULT_CANDIDATE_TAG,
            DEFAULT_DROP_FIRST,
            DEFAULT_ORACLE_TAG,
            compute_tac_from_best_run_json,
            compute_tac_from_log_dir,
        )
        import isaaclab_eureka.tac_score as mod  # type: ignore
        return mod

    spec = importlib.util.spec_from_file_location("_tacreka_tac_score", candidate)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_tac = _load_tac_score_module()
DEFAULT_ORACLE_TAG = _tac.DEFAULT_ORACLE_TAG
DEFAULT_CANDIDATE_TAG = _tac.DEFAULT_CANDIDATE_TAG
DEFAULT_DROP_FIRST = _tac.DEFAULT_DROP_FIRST
compute_tac_from_best_run_json = _tac.compute_tac_from_best_run_json
compute_tac_from_log_dir = _tac.compute_tac_from_log_dir


def _resolve_input(path: str, oracle_tag: str, candidate_tag: str, drop_first: int) -> dict:
    """Dispatch to the right helper based on whether ``path`` is a JSON file or a log dir."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input not found: {path}")
    if os.path.isfile(path) and path.endswith(".json"):
        return compute_tac_from_best_run_json(
            path,
            oracle_tag=oracle_tag,
            candidate_tag=candidate_tag,
            drop_first=drop_first,
        )
    if os.path.isdir(path):
        return compute_tac_from_log_dir(
            path,
            oracle_tag=oracle_tag,
            candidate_tag=candidate_tag,
            drop_first=drop_first,
        )
    raise ValueError(f"Don't know how to interpret input {path!r}; expected a .json or a directory.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the Trajectory Alignment Coefficient (Kendall's Tau-b) between the"
            " candidate (LLM) reward and the oracle (task) reward for one or more Tacreka"
            " training runs, using the per-iteration scalars already logged to TensorBoard."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        action="append",
        required=True,
        help=(
            "A path to either (a) a Tacreka `best_run.json` (its `training_log_dir` is used)"
            " or (b) an rsl_rl run directory containing TensorBoard event files. Pass"
            " --input multiple times to score several runs at once."
        ),
    )
    parser.add_argument(
        "--oracle_tag",
        type=str,
        default=DEFAULT_ORACLE_TAG,
        help="TensorBoard tag (suffix-matched) for the oracle expected return per iteration.",
    )
    parser.add_argument(
        "--candidate_tag",
        type=str,
        default=DEFAULT_CANDIDATE_TAG,
        help="TensorBoard tag (suffix-matched) for the candidate expected return per iteration.",
    )
    parser.add_argument(
        "--drop_first",
        type=int,
        default=DEFAULT_DROP_FIRST,
        help="Drop this many initial scalar points before computing the rank correlation.",
    )
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    results = []
    for raw in args.input:
        try:
            res = _resolve_input(raw, args.oracle_tag, args.candidate_tag, args.drop_first)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"[TAC] {raw}: ERROR -- {exc}", file=sys.stderr)
            results.append({"input": raw, "error": str(exc)})
            continue
        res["input"] = raw
        tau = res["tac_score_kendall_tau_b"]
        p = res["tac_p_value"]
        n = res["n_eta"]
        rho = res["spearman_rho"]
        pearson = res["pearson_correlation"]
        print(
            f"[TAC] {raw}\n"
            f"       n_eta={n}  tau_b={tau:+.4f}  p={p:.3g}"
            f"  rho={rho:+.4f}  pearson={pearson:+.4f}"
        )
        results.append(res)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"results": results}, f, indent=2)
        print(f"Wrote TAC results to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
