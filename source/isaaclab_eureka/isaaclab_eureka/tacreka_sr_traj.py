# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

"""Tacreka SR + offline trajectory recording.

Thin subclass of :class:`isaaclab_eureka.tacreka_sr_testing.Tacreka_SR` that, *after the
entire Tacreka_SR training session is done*, invokes
``scripts/record_offline_trajectories.py`` on every successful training run's checkpoint
directory. The recorder discovers ``model_<iter>.pt`` files at every ``record_every``
(default 100) PPO iterations and saves per-step state + oracle/candidate rewards as an
offline dataset that can later be used to evaluate any new candidate reward without
re-training.

Why post-training (subprocess after `_task_manager.close()`)?

  * Recording mid-training would mean two AppLaunchers / PhysX contexts on the GPU at
    once -- on an 8 GB card that OOMs while allocating contact-pair buffers
    (``Failed to allocate ... for mGpuContactPairsDev``). The training-side worker holds
    the env alive between Eureka iterations, so even *between* iters the GPU is busy.
  * Loading a checkpoint and rolling out gives statistically the same trajectory
    distribution ``eta_t`` as recording mid-training (both share the env's reset
    distribution mu); post-hoc rollout actually starts from a *clean* mu rather than
    the biased mid-training rollout buffer.
  * It keeps the training pipeline untouched; the recorder is just a separate script.

So the flow is:

  1. During each Eureka iteration, ``_log_iteration_results`` records nothing -- it just
     queues a ``(iter, run_idx, run_dir, reward_fn_str)`` tuple.
  2. After ``super().run(...)`` returns (which has already called
     ``self._task_manager.close()`` and freed the GPU), we drain the queue and invoke
     the recorder for each queued run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from isaaclab_eureka.tacreka_sr_testing import Tacreka_SR


def _find_recorder_script() -> str:
    """Locate ``scripts/record_offline_trajectories.py`` relative to the workspace root."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        # Tacreka/source/isaaclab_eureka/isaaclab_eureka/ -> Tacreka/scripts/
        os.path.normpath(os.path.join(here, "..", "..", "..", "scripts", "record_offline_trajectories.py")),
        # workspace root sibling
        os.path.normpath(os.path.join(here, "..", "..", "..", "..", "Tacreka", "scripts", "record_offline_trajectories.py")),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"record_offline_trajectories.py not found. Tried:\n  - " + "\n  - ".join(candidates)
    )


class Tacreka_SR_Traj(Tacreka_SR):
    """:class:`Tacreka_SR` plus an offline trajectory recording step after every successful run.

    Args (in addition to :class:`Tacreka_SR`):
        record_every: Sub-sample ``model_*.pt`` checkpoints every N PPO iterations
            (default 100).
        record_num_episodes: How many complete episodes to roll out per snapshot.
        record_num_envs: Parallel envs to use during recording rollouts.
        record_gamma: Discount factor for the saved ``episode_*_returns`` summaries
            (per-step rewards are stored regardless, so any gamma can be re-applied later).
        record_only_best_per_iter: If True, only record the best-performing run within
            each Eureka iteration. If False (default), record every successful run.
        record_max_total_steps: Per-snapshot safety cap on env-steps.
        record_in_background: If True, fire-and-forget the recorder subprocess so the
            next Eureka iteration can start training immediately. If False (default),
            block until recording completes.
    """

    def __init__(
        self,
        task: str,
        *args,
        record_every: int = 100,
        record_num_episodes: int = 8,
        record_num_envs: int = 8,
        record_gamma: float = 1.0,
        record_only_best_per_iter: bool = False,
        record_max_total_steps: int = 20_000,
        record_in_background: bool = False,
        **kwargs,
    ):
        super().__init__(task, *args, **kwargs)
        self._task = task
        self._record_every = int(record_every)
        self._record_num_episodes = int(record_num_episodes)
        self._record_num_envs = int(record_num_envs)
        self._record_gamma = float(record_gamma)
        self._record_only_best_per_iter = bool(record_only_best_per_iter)
        self._record_max_total_steps = int(record_max_total_steps)
        self._record_in_background = bool(record_in_background)

        try:
            self._recorder_script = _find_recorder_script()
        except FileNotFoundError as err:
            print(f"[WARN][Tacreka_SR_Traj] {err}. Recording will be skipped.")
            self._recorder_script = None

        self._recorder_dir = os.path.join(self._log_dir, "offline_trajectories")
        os.makedirs(self._recorder_dir, exist_ok=True)

        # queue of pending recordings; drained after super().run() returns (GPU free).
        # each item: dict(iter, run_idx, run_dir, reward_function, log_dir)
        self._pending_recordings: list[dict] = []
        self._bg_recordings: list[subprocess.Popen] = []

    # ------------------------------------------------------------------
    # Recorder invocation
    # ------------------------------------------------------------------

    def _invoke_recorder(self, payload_path: str, output_dir: str) -> None:
        """Run the recorder script as a subprocess (blocking unless background mode)."""
        if not self._recorder_script:
            return

        cmd = [
            sys.executable,
            self._recorder_script,
            "--best_run_json", payload_path,
            "--task", self._task,
            "--every", str(self._record_every),
            "--num_episodes", str(self._record_num_episodes),
            "--num_envs", str(self._record_num_envs),
            "--gamma", str(self._record_gamma),
            "--max_total_steps", str(self._record_max_total_steps),
            "--output_dir", output_dir,
        ]
        print(f"[Tacreka_SR_Traj] Launching recorder: {' '.join(cmd)}")

        log_path = os.path.join(output_dir, "recorder_stdout.log")
        log_handle = open(log_path, "w")

        if self._record_in_background:
            proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT)
            self._bg_recordings.append(proc)
            print(f"[Tacreka_SR_Traj]   (background pid={proc.pid}, log={log_path})")
        else:
            try:
                subprocess.run(cmd, stdout=log_handle, stderr=subprocess.STDOUT, check=True)
                print(f"[Tacreka_SR_Traj]   recording done -> {output_dir}")
            except subprocess.CalledProcessError as err:
                print(f"[Tacreka_SR_Traj]   recorder failed: {err}. See {log_path}.")
            finally:
                log_handle.close()

    def _queue_run(self, iter_idx: int, run_idx: int, result: dict) -> None:
        """Add a successful run to the pending-recordings queue (no subprocess yet)."""
        run_dir = result.get("run_dir") or result.get("log_dir")
        if not run_dir or not os.path.isdir(run_dir):
            print(f"[Tacreka_SR_Traj] iter={iter_idx} run={run_idx}: no valid run_dir, skipping.")
            return
        reward_fn = result.get("reward_function")
        if not reward_fn:
            print(f"[Tacreka_SR_Traj] iter={iter_idx} run={run_idx}: no reward_function, skipping.")
            return
        self._pending_recordings.append({
            "iter": iter_idx,
            "run_idx": run_idx,
            "run_dir": run_dir,
            "log_dir": result.get("log_dir"),
            "reward_function": reward_fn,
        })

    def _drain_pending_recordings(self) -> None:
        """Run the recorder for every queued run. Called only after the GPU has been freed
        by the parent's ``self._task_manager.close()``."""
        if not self._pending_recordings:
            print("[Tacreka_SR_Traj] No pending recordings to process.")
            return
        if not self._recorder_script:
            print("[Tacreka_SR_Traj] No recorder script available; skipping all recordings.")
            return

        print(
            f"[Tacreka_SR_Traj] Training done. Draining {len(self._pending_recordings)} "
            "queued recordings now that the GPU is free."
        )
        for item in self._pending_recordings:
            iter_idx = item["iter"]
            run_idx = item["run_idx"]
            out_dir = os.path.join(self._recorder_dir, f"iter_{iter_idx:03d}_run_{run_idx:02d}")
            os.makedirs(out_dir, exist_ok=True)
            payload_path = os.path.join(out_dir, "_recorder_input.json")
            with open(payload_path, "w") as f:
                json.dump(
                    {
                        "training_run_dir": item["run_dir"],
                        "training_log_dir": item["log_dir"],
                        "reward_function": item["reward_function"],
                    },
                    f,
                    indent=2,
                )
            self._invoke_recorder(payload_path, out_dir)

        # If anything was launched in the background, wait now.
        if self._bg_recordings:
            print(f"[Tacreka_SR_Traj] Waiting for {len(self._bg_recordings)} background recordings...")
            for proc in self._bg_recordings:
                proc.wait()
            print("[Tacreka_SR_Traj] All background recordings done.")

    # ------------------------------------------------------------------
    # Hook into the parent training loop -- queue only, do not subprocess.
    # ------------------------------------------------------------------

    def _log_iteration_results(self, iter, results):
        # 1) parent's logging / TensorBoard / wandb writes
        super()._log_iteration_results(iter, results)

        # 2) queue successful runs for later recording
        if self._record_only_best_per_iter:
            best = None
            best_idx = None
            for idx, result in enumerate(results):
                if not result.get("success"):
                    continue
                metric = result.get("success_metric_max")
                if metric is None:
                    continue
                if best is None or (
                    self._success_metric_to_win is not None
                    and abs(metric - self._success_metric_to_win)
                    < abs(best - self._success_metric_to_win)
                ):
                    best = metric
                    best_idx = idx
            if best_idx is not None:
                self._queue_run(iter, best_idx, results[best_idx])
        else:
            for idx, result in enumerate(results):
                if result.get("success"):
                    self._queue_run(iter, idx, result)

    # ------------------------------------------------------------------
    # Override run(): after the parent's run() returns the GPU is free
    # (parent calls self._task_manager.close() at the end), so this is
    # the right moment to actually record trajectories.
    # ------------------------------------------------------------------

    def run(self, *args, **kwargs):
        result = super().run(*args, **kwargs)
        try:
            self._drain_pending_recordings()
        except Exception as err:  # don't let recording failures hide the training result
            print(f"[Tacreka_SR_Traj] Recording phase raised: {err!r}")
        return result
