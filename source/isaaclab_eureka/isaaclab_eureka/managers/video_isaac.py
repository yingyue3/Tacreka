"""VideoIsaac — self-contained class for recording Isaac Lab RSL-RL policy clips.

The class manages the full Isaac Sim lifecycle and exposes a
``record(checkpoint, output_dir)`` entry point.  The gymnasium environment
is created **once** on the first ``record()`` call and reused for every
subsequent call — only the policy weights are swapped out.  Recording is
driven manually via ``RecordVideo``'s public ``start_recording`` /
``stop_recording`` API, so a fresh MP4 is produced for each call.

All heavy Isaac / rsl_rl imports are deferred until after ``AppLauncher`` has
started the simulator, so the module is safe to import at any time.

**Standalone usage** (class starts the simulator)::

    recorder = VideoIsaac(
        task="Isaac-Lift-Cube-Franka-v0",
        num_clips=1,
        clip_length=200,
        headless=True,
    )
    recorder.record("/path/to/model_A.pt", "./recordings/run_A.mp4")
    recorder.record("/path/to/model_B.pt", "./recordings/run_B.mp4")
    recorder.close()

As a context manager::

    with VideoIsaac(task="Isaac-Lift-Cube-Franka-v0") as recorder:
        recorder.record("/path/to/model.pt", "./recordings/run.mp4")

**Embedded** in a script that already launched ``AppLauncher``::

    recorder = VideoIsaac(task=task, simulation_app=simulation_app)
    recorder.record(checkpoint, "./recordings/run.mp4")

``output_dir`` is the desired output **file** path.  The parent directory is
created automatically.  For multiple clips the stem is suffixed::

    ./recordings/run.mp4             # single clip  (num_clips=1)
    ./recordings/run_clip0.mp4       # multiple clips
    ./recordings/run_clip1.mp4
"""

from __future__ import annotations

import os
import shutil
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg


class VideoIsaac:
    """Records video clips from a trained Isaac Lab RSL-RL policy.

    The gymnasium environment (and its ``RecordVideo`` wrapper) is created
    once and reused across all ``record()`` calls so that Isaac Sim's scene
    does not need to be rebuilt for each checkpoint.

    Args:
        task: Gymnasium task ID (e.g. ``"Isaac-Lift-Cube-Franka-v0"``).
        num_envs: Number of parallel environments.  Use ``1`` for recording.
        device: Torch / Isaac device string (e.g. ``"cuda:0"``).
        headless: Run without a display.  Should be ``True`` for batch
            recording.
        num_clips: Default number of clips per ``record()`` call.
        clip_length: Default steps per clip.
        real_time: Throttle simulation steps to wall-clock real time.
        agent_cfg_entry_point: Gym registry key used to load the agent
            config.  Defaults to ``"rsl_rl_cfg_entry_point"``.
        simulation_app: An already-running ``SimulationApp`` instance.
            When ``None`` (default), ``VideoIsaac`` launches
            ``AppLauncher`` itself and owns the simulator lifecycle.
    """

    def __init__(
        self,
        task: str,
        num_envs: int = 1,
        device: str = "cuda:0",
        headless: bool = True,
        num_clips: int = 1,
        clip_length: int = 200,
        real_time: bool = False,
        agent_cfg_entry_point: str = "rsl_rl_cfg_entry_point",
        simulation_app: Any = None,
    ) -> None:
        self.task = task
        self.num_envs = num_envs
        self.device = device
        self.num_clips = num_clips
        self.clip_length = clip_length
        self.real_time = real_time
        self.agent_cfg_entry_point = agent_cfg_entry_point

        # -- simulator -------------------------------------------------------
        if simulation_app is None:
            from isaaclab.app import AppLauncher

            launcher_args = {"headless": headless, "enable_cameras": True}
            print(f"[VideoIsaac] Launching Isaac Sim (headless={headless}) ...")
            self._app_launcher = AppLauncher(launcher_args)
            self._simulation_app = self._app_launcher.app
            self._owns_sim = True
            print("[VideoIsaac] Isaac Sim is running.")
        else:
            self._app_launcher = None
            self._simulation_app = simulation_app
            self._owns_sim = False

        # -- env state (populated on first record() call) --------------------
        self._env: Any = None           # RslRlVecEnvWrapper
        self._record_env: Any = None    # gym.wrappers.RecordVideo
        self._agent_cfg: Any = None
        self._staging_dir: str | None = None
        # Monotonic counter used to give every recorded clip a unique staging
        # filename, so successive record() calls don't clobber each other.
        self._clip_counter: int = 0

    # ------------------------------------------------------------------
    # Environment construction (runs once)
    # ------------------------------------------------------------------

    def _build_env(self, staging_dir: str) -> None:
        """Create the gymnasium env with a ``RecordVideo`` wrapper.

        Called automatically on the first ``record()`` invocation.  All
        Isaac / rsl_rl imports happen here (after the simulator is up).

        The ``step_trigger`` is set to ``lambda _: False`` and ``video_length``
        to ``0`` (treated by gymnasium as "unlimited") — recording is driven
        manually via ``start_recording`` / ``stop_recording`` in ``record()``.

        Args:
            staging_dir: Directory where ``RecordVideo`` writes raw MP4 files
                before ``record()`` moves them to the user-supplied path.
        """
        import gymnasium as gym
        from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg
        import isaaclab_tasks  # noqa: F401

        os.makedirs(staging_dir, exist_ok=True)
        self._staging_dir = staging_dir

        env_cfg = parse_env_cfg(self.task, device=self.device, num_envs=self.num_envs)
        self._agent_cfg = load_cfg_from_registry(self.task, self.agent_cfg_entry_point)

        print(f"[VideoIsaac] Creating environment: {self.task}")
        raw_env = gym.make(self.task, cfg=env_cfg, render_mode="rgb_array")

        if isinstance(raw_env.unwrapped, DirectMARLEnv):
            raw_env = multi_agent_to_single_agent(raw_env)

        # Auto-triggers are disabled; recording is driven manually via the
        # RecordVideo.start_recording / stop_recording API in record().
        print(f"[VideoIsaac] Attaching RecordVideo wrapper (staging_dir={staging_dir})")
        self._record_env = gym.wrappers.RecordVideo(
            raw_env,
            video_folder=staging_dir,
            step_trigger=lambda _: False,
            video_length=0,
            disable_logger=True,
        )
        self._env = RslRlVecEnvWrapper(self._record_env, clip_actions=self._agent_cfg.clip_actions)
        print(f"[VideoIsaac] Environment ready. Staging dir: {staging_dir}")

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        checkpoint: str,
        output_file: str,
        num_clips: int | None = None,
        clip_length: int | None = None,
    ) -> list[str]:
        """Load *checkpoint*, run the policy, and save video clips.

        The environment is created on the first call and reused on every
        subsequent call — only the policy weights are swapped.

        Args:
            checkpoint: Absolute or relative path to a ``.pt`` model checkpoint.
            output_dir: Desired output file path
                (e.g. ``"./recordings/run.mp4"``).  The parent directory is
                created automatically.  For multiple clips the stem is
                suffixed: ``run_clip0.mp4``, ``run_clip1.mp4``, …
            num_clips: Number of consecutive clips to capture.  Overrides
                the value set in ``__init__`` for this call only.
            clip_length: Simulation steps per clip.  Overrides the value set
                in ``__init__`` for this call only.

        Returns:
            List of absolute paths to the saved MP4 files (in clip order).
        """
        import torch
        from isaaclab.utils.assets import retrieve_file_path
        from rsl_rl.runners import DistillationRunner, OnPolicyRunner

        n_clips = num_clips if num_clips is not None else self.num_clips
        c_length = clip_length if clip_length is not None else self.clip_length

        checkpoint = os.path.abspath(checkpoint)
        output_dir = os.path.abspath(output_file)
        output_stem, output_ext = os.path.splitext(output_dir)
        if not output_ext:
            output_ext = ".mp4"
        parent_dir = os.path.dirname(output_dir)
        os.makedirs(parent_dir, exist_ok=True)

        # Build the env once; on subsequent calls it is already ready.
        if self._env is None:
            self._build_env(os.path.join(parent_dir, "_video_isaac_tmp"))

        resume_path = retrieve_file_path(checkpoint)
        print(f"[VideoIsaac] Loading checkpoint: {resume_path}")

        if self._agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(
                self._env, self._agent_cfg.to_dict(), log_dir=None, device=self._agent_cfg.device
            )
        elif self._agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(
                self._env, self._agent_cfg.to_dict(), log_dir=None, device=self._agent_cfg.device
            )
        else:
            raise ValueError(f"Unsupported runner class: {self._agent_cfg.class_name!r}")

        runner.load(resume_path)
        policy = runner.get_inference_policy(device=self._env.unwrapped.device)

        try:
            policy_nn = runner.alg.policy        # rsl_rl >= 2.3
        except AttributeError:
            policy_nn = runner.alg.actor_critic  # rsl_rl <= 2.2

        dt = self._env.unwrapped.step_dt
        print(
            f"\n[VideoIsaac] Recording: {n_clips} clip(s) × {c_length} steps "
            f"= {n_clips * c_length} total steps"
        )

        obs = self._env.get_observations()
        saved: list[str] = []

        for clip_idx in range(n_clips):
            if not self._simulation_app.is_running():
                break

            # Unique staging filename — RecordVideo appends ".mp4".
            staging_name = f"_clip_{self._clip_counter:05d}"
            staging_mp4 = os.path.join(self._staging_dir, f"{staging_name}.mp4")
            self._clip_counter += 1

            print(f"[VideoIsaac]   clip {clip_idx + 1}/{n_clips}: start_recording({staging_name!r})")
            self._record_env.start_recording(staging_name)

            obs = self._run_policy_for_steps(obs, policy, policy_nn, c_length, dt)

            # Stop and flush MP4 to disk.
            self._record_env.stop_recording()

            if not os.path.isfile(staging_mp4):
                print(f"[VideoIsaac]   warning: expected {staging_mp4} but it was not written.")
                continue

            suffix = f"_clip{clip_idx}" if n_clips > 1 else ""
            dst = f"{output_stem}{suffix}{output_ext}"
            shutil.move(staging_mp4, dst)
            print(f"[VideoIsaac]   saved: {dst}")
            saved.append(dst)

        print(f"[VideoIsaac] Done. {len(saved)} clip(s) saved.")
        return saved

    def _run_policy_for_steps(
        self,
        obs: Any,
        policy: Any,
        policy_nn: Any,
        num_steps: int,
        dt: float,
    ) -> Any:
        """Step the env *num_steps* times and return the final observation."""
        import torch

        timestep = 0
        while self._simulation_app.is_running() and timestep < num_steps:
            start_time = time.time()

            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = self._env.step(actions)
                policy_nn.reset(dones)

            timestep += 1

            if self.real_time:
                sleep_time = dt - (time.time() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        return obs

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the environment and (if owned) shut down Isaac Sim."""
        if self._env is not None:
            print("[VideoIsaac] Closing environment ...")
            self._env.close()
            self._env = None
            self._record_env = None
        if self._staging_dir and os.path.isdir(self._staging_dir):
            shutil.rmtree(self._staging_dir, ignore_errors=True)
        if self._owns_sim and self._simulation_app is not None:
            if self._simulation_app.is_running():
                print("[VideoIsaac] Closing Isaac Sim ...")
                self._simulation_app.close()

    def __enter__(self) -> VideoIsaac:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
