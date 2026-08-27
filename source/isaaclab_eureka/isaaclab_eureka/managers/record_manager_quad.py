# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

"""Record Quadcopter policy rollouts without Isaac camera rendering.

This module provides a RecordManagerQuadcopter class that runs policy inference
headlessly and generates a lightweight 2D MP4 visualization from simulation
states and observations. It is intended for clusters where Isaac renderer-based
recording is unavailable.
"""

from __future__ import annotations

import math
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from isaaclab_eureka.utils import resolve_sim_device

# State fields snapshot by ``_capture_state``. Mirrors the schema used by
# ``scripts/record_offline_trajectories.py`` so saved trajectories can be replayed
# by ``scripts/eval_reward_on_offline_dataset.py`` and the human-preference TAC
# script without any per-task adapter code.
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
            if tensor is not None and hasattr(tensor, "detach"):
                out[name] = tensor.detach().clone().cpu()
    for name in _ENV_FIELDS:
        tensor = getattr(base_env, name, None)
        if tensor is not None and hasattr(tensor, "detach"):
            out[name] = tensor.detach().clone().cpu()
    return out

# import gymnasium as gym
# import isaaclab_tasks  # noqa: F401
# from isaaclab.envs import DirectRLEnvCfg
# from isaaclab_tasks.utils import parse_env_cfg
# from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


class RecordManagerQuad:
    """Manager for recording quadcopter policy rollouts with fallback 2D rendering."""

    def __init__(
        self,
        task: str,
        num_envs: int = 1,
        device: str = "cuda",
        rl_library: str = "rsl_rl",
        headless: bool = True,
        # output_file: str = "./recordings/quadcopter_fallback.mp4",
        fps: int = 30,
        max_frames: int = 900,
        num_episodes: int = 1,
        env_index: int = 0,
        trail_length: int = 160,
        frame_width: int = 1280,
        frame_height: int = 720,
        save_trajectory: bool = True,
        trajectory_gamma: float = 1.0,
    ):
        """Initialize the quadcopter recording manager.

        Args:
            task: Name of the task/environment.
            checkpoint: Path to model checkpoint.
            num_envs: Number of environments to simulate.
            device: Device for simulation and policy inference.
            rl_library: RL library used to train the checkpoint ("rsl_rl" or "rl_games").
            headless: Force display off at all times.
            output_file: Output MP4 path.
            fps: Video FPS.
            max_frames: Maximum frames to write. <=0 means unlimited.
            num_episodes: Stop after this many completed episodes. <=0 means unlimited.
            env_index: Environment index to visualize.
            trail_length: How many recent positions to draw as trajectory.
            frame_width: Video frame width in pixels.
            frame_height: Video frame height in pixels.
            save_trajectory: If True, also save the per-step trajectory used to
                generate the video as a ``.pt`` file alongside the MP4. The schema
                matches ``record_offline_trajectories.py`` so that the saved file is
                directly consumable by ``eval_reward_on_offline_dataset.py``.
            trajectory_gamma: Discount factor for the per-episode return summaries
                stored in the trajectory file. Per-step rewards are saved verbatim
                so any other gamma can be applied later.
        """
        self.task = task
        # self.checkpoint = checkpoint
        self.num_envs = num_envs
        self.device = device
        self.rl_library = rl_library
        self.headless = headless
        # self.output_file = output_file
        self.fps = fps
        self.max_frames = max_frames
        self.num_episodes = num_episodes
        self.env_index = env_index
        self.trail_length = trail_length
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.save_trajectory = bool(save_trajectory)
        self.trajectory_gamma = float(trajectory_gamma)

        self._frame_idx = 0
        self._episode_idx = 0
        self._episode_step = 0
        self._yaw_est = 0.0
        self._trail_xy: list[tuple[float, float]] = []
        self._trail_xz: list[tuple[float, float]] = []
        self._fallback_state = {"pseudo_pos": np.zeros(3, dtype=float)}
        self._step_dt = 0.01

        # Trajectory recording buffers, populated by ``_traj_*`` helpers during
        # ``record()`` when ``self.save_trajectory`` is True. Schema mirrors
        # ``scripts/record_offline_trajectories.py`` exactly so the saved ``.pt`` files
        # can be consumed by ``scripts/eval_reward_on_offline_dataset.py`` (and the
        # human-preference TAC script) with no special-casing.
        self._traj_step_obs: list = []
        self._traj_step_action: list = []
        self._traj_step_oracle: list = []
        self._traj_step_done: list = []
        self._traj_step_episode_id: list = []
        self._traj_step_state: dict[str, list] = {}
        self._traj_episode_id = None  # torch.LongTensor[num_envs], lazily initialized
        self._traj_episode_oracle_returns: list[float] = []
        self._traj_oracle_return_buf = None
        self._traj_discount_buf = None
        self._traj_active: bool = False
        self._traj_n_episodes_completed: int = 0
        self._traj_checkpoint_path: str | None = None

        import torch

        from isaaclab.app import AppLauncher

        device = resolve_sim_device(self.device)

        app_launcher = AppLauncher(headless=self.headless, device=device)
        simulation_app = app_launcher.app
        self.simulation_app = simulation_app
        self.device = device  # update to resolved device (e.g. "cuda:0")

        import gymnasium as gym

        import isaaclab_tasks  # noqa: F401
        from isaaclab.envs import DirectRLEnvCfg
        from isaaclab_tasks.utils import parse_env_cfg
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        env_cfg: DirectRLEnvCfg = parse_env_cfg(self.task)
        env_cfg.sim.device = device
        env_cfg.scene.num_envs = self.num_envs if self.num_envs is not None else 1
        self.env_cfg = env_cfg
        self.env = gym.make(self.task, cfg=env_cfg)


    @staticmethod
    def _extract_policy_obs(obs):
        """Extract policy observations tensor from TensorDict/dict/tensor-like outputs."""
        if hasattr(obs, "keys"):
            keys = set(obs.keys())
            if "policy" in keys:
                return obs["policy"]
            if "obs" in keys:
                inner_obs = obs["obs"]
                if isinstance(inner_obs, dict):
                    if "policy" in inner_obs:
                        return inner_obs["policy"]
                    return next(iter(inner_obs.values()))
                return inner_obs
        return obs

    @staticmethod
    def _project_to_panel(
        x: float,
        y: float,
        bounds: tuple[float, float, float, float],
        panel: tuple[int, int, int, int],
    ) -> tuple[int, int]:
        """Project world coordinates to panel pixel coordinates."""
        x_min, x_max, y_min, y_max = bounds
        px0, py0, px1, py1 = panel
        span_x = max(1e-6, x_max - x_min)
        span_y = max(1e-6, y_max - y_min)
        px = px0 + (x - x_min) / span_x * (px1 - px0)
        py = py1 - (y - y_min) / span_y * (py1 - py0)
        return int(px), int(py)

    @staticmethod
    def _compute_bounds_2d(
        points: list[tuple[float, float]],
        min_span: float = 4.0,
        margin_ratio: float = 0.15,
    ) -> tuple[float, float, float, float]:
        """Compute bounding box for 2D points with margin."""
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)

        span_x = max(min_span, x_max - x_min)
        span_y = max(min_span, y_max - y_min)
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)

        x_half = 0.5 * span_x * (1.0 + margin_ratio)
        y_half = 0.5 * span_y * (1.0 + margin_ratio)
        return cx - x_half, cx + x_half, cy - y_half, cy + y_half

    def _draw_trail(
        self,
        draw: ImageDraw.ImageDraw,
        trail: list[tuple[float, float]],
        bounds: tuple[float, float, float, float],
        panel: tuple[int, int, int, int],
        color: tuple[int, int, int],
    ):
        """Draw trajectory trail on the panel."""
        if len(trail) < 2:
            return
        points = [self._project_to_panel(x, y, bounds, panel) for (x, y) in trail]
        draw.line(points, fill=color, width=3)

    def _draw_drone_topdown(
        self,
        draw: ImageDraw.ImageDraw,
        x: float,
        y: float,
        yaw: float,
        bounds: tuple[float, float, float, float],
        panel: tuple[int, int, int, int],
    ):
        """Draw drone icon in top-down view."""
        cx, cy = self._project_to_panel(x, y, bounds, panel)
        r = 8
        draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            fill=(38, 132, 255),
            outline=(10, 60, 120),
            width=2,
        )

        arm = 18
        fx = cx + int(arm * math.cos(yaw))
        fy = cy - int(arm * math.sin(yaw))
        draw.line([(cx, cy), (fx, fy)], fill=(10, 60, 120), width=3)

    def _draw_goal(
        self,
        draw: ImageDraw.ImageDraw,
        x: float,
        y: float,
        bounds: tuple[float, float, float, float],
        panel: tuple[int, int, int, int],
    ):
        """Draw goal marker on the panel."""
        gx, gy = self._project_to_panel(x, y, bounds, panel)
        s = 8
        draw.line([(gx - s, gy), (gx + s, gy)], fill=(204, 53, 53), width=3)
        draw.line([(gx, gy - s), (gx, gy + s)], fill=(204, 53, 53), width=3)
        draw.ellipse(
            [gx - s - 3, gy - s - 3, gx + s + 3, gy + s + 3],
            outline=(204, 53, 53),
            width=2,
        )

    def _render_frame(
        self,
        pos_w: np.ndarray,
        desired_w: np.ndarray,
        lin_vel_b: np.ndarray,
        ang_vel_b: np.ndarray,
        reward: float,
        done: bool,
        distance_to_goal: float,
    ) -> np.ndarray:
        """Render a single visualization frame."""
        image = Image.new("RGB", (self.frame_width, self.frame_height), (246, 247, 250))
        draw = ImageDraw.Draw(image)

        left_panel = (30, 70, self.frame_width // 2 - 20, self.frame_height - 40)
        right_panel = (self.frame_width // 2 + 20, 70, self.frame_width - 30, self.frame_height - 40)

        draw.rectangle(left_panel, outline=(86, 94, 106), width=2)
        draw.rectangle(right_panel, outline=(86, 94, 106), width=2)
        draw.text((left_panel[0], 30), "Top-down (x-y)", fill=(30, 30, 30))
        draw.text((right_panel[0], 30), "Side view (x-z)", fill=(30, 30, 30))

        xy_bounds = self._compute_bounds_2d(
            [(pos_w[0], pos_w[1]), (desired_w[0], desired_w[1])]
            + self._trail_xy[-30:],
            min_span=4.0,
            margin_ratio=0.2,
        )
        z_bounds = self._compute_bounds_2d(
            [(pos_w[0], pos_w[2]), (desired_w[0], desired_w[2])]
            + self._trail_xz[-30:],
            min_span=2.5,
            margin_ratio=0.2,
        )

        self._draw_trail(draw, self._trail_xy, xy_bounds, left_panel, (102, 157, 246))
        self._draw_goal(draw, desired_w[0], desired_w[1], xy_bounds, left_panel)
        self._draw_drone_topdown(draw, pos_w[0], pos_w[1], self._yaw_est, xy_bounds, left_panel)

        self._draw_trail(draw, self._trail_xz, z_bounds, right_panel, (102, 157, 246))
        self._draw_goal(draw, desired_w[0], desired_w[2], z_bounds, right_panel)
        dx, dz = self._project_to_panel(pos_w[0], pos_w[2], z_bounds, right_panel)
        r = 8
        draw.ellipse(
            [dx - r, dz - r, dx + r, dz + r],
            fill=(38, 132, 255),
            outline=(10, 60, 120),
            width=2,
        )

        status = "DONE" if done else "RUN"
        status_color = (173, 25, 25) if done else (32, 32, 32)
        text_lines = [
            f"status={status} frame={self._frame_idx} episode={self._episode_idx} step={self._episode_step}",
            f"pos_w=({pos_w[0]:+.2f}, {pos_w[1]:+.2f}, {pos_w[2]:+.2f})",
            f"goal_w=({desired_w[0]:+.2f}, {desired_w[1]:+.2f}, {desired_w[2]:+.2f}) dist={distance_to_goal:.3f}",
            f"lin_vel_b=({lin_vel_b[0]:+.2f}, {lin_vel_b[1]:+.2f}, {lin_vel_b[2]:+.2f})",
            f"ang_vel_b=({ang_vel_b[0]:+.2f}, {ang_vel_b[1]:+.2f}, {ang_vel_b[2]:+.2f}) reward={reward:+.3f}",
        ]
        y = 8
        for line in text_lines:
            draw.text((20, y), line, fill=status_color)
            y += 18

        return np.asarray(image, dtype=np.uint8)

    def _read_quadcopter_state(
        self,
        base_env,
        env_index: int,
        obs_row: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Read quadcopter state from environment or fallback to observation-based estimation."""
        pos_w = None
        desired_w = None

        if hasattr(base_env, "_robot") and hasattr(base_env._robot, "data"):
            try:
                pos_w = base_env._robot.data.root_pos_w[env_index].detach().cpu().numpy()
            except Exception:
                pos_w = None

        if hasattr(base_env, "_desired_pos_w"):
            try:
                desired_w = base_env._desired_pos_w[env_index].detach().cpu().numpy()
            except Exception:
                desired_w = None

        lin_vel_b = np.array(obs_row[0:3], dtype=float)
        ang_vel_b = np.array(obs_row[3:6], dtype=float)
        desired_pos_b = (
            np.array(obs_row[9:12], dtype=float)
            if obs_row.shape[0] >= 12
            else np.zeros(3, dtype=float)
        )

        if pos_w is None:
            self._fallback_state["pseudo_pos"] = (
                self._fallback_state["pseudo_pos"] + lin_vel_b * self._step_dt
            )
            pos_w = self._fallback_state["pseudo_pos"].copy()
        if desired_w is None:
            desired_w = pos_w + desired_pos_b

        return pos_w, desired_w, lin_vel_b, ang_vel_b

    def _reset_episode_state(self):
        """Reset state for a new episode."""
        self._episode_step = 0
        self._yaw_est = 0.0
        self._trail_xy.clear()
        self._trail_xz.clear()
        self._fallback_state["pseudo_pos"] = np.zeros(3, dtype=float)

    def _update_trails(self, pos_w: np.ndarray):
        """Update trajectory trails with new position."""
        self._trail_xy.append((float(pos_w[0]), float(pos_w[1])))
        self._trail_xz.append((float(pos_w[0]), float(pos_w[2])))
        if len(self._trail_xy) > self.trail_length:
            self._trail_xy = self._trail_xy[-self.trail_length:]
            self._trail_xz = self._trail_xz[-self.trail_length:]

    def _should_stop(self) -> bool:
        """Check if recording should stop."""
        if self.max_frames > 0 and self._frame_idx >= self.max_frames:
            return True
        if self.num_episodes > 0 and self._episode_idx >= self.num_episodes:
            return True
        return False

    def record(self, output_file: str, checkpoint: str, trajectory_output_file: str | None = None):
        """Run policy rollout and record video.

        Args:
            output_file: Path to the output MP4 video file.
            checkpoint: Path to the policy checkpoint to load.
            trajectory_output_file: Optional path for the saved trajectory ``.pt`` file.
                If ``self.save_trajectory`` is True and this is None, defaults to
                ``<output_file>.pt`` (so ``foo.mp4`` -> ``foo.pt``).
        """
        import torch

        import gymnasium as gym

        import isaaclab_tasks  # noqa: F401
        from isaaclab.envs import DirectRLEnvCfg
        from isaaclab_tasks.utils import parse_env_cfg
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        writer = imageio.get_writer(output_file, fps=self.fps)

        active_env_idx = max(0, self.env_index)
        self._frame_idx = 0
        self._episode_idx = 0
        self._episode_step = 0
        self._step_dt = float(getattr(self.env_cfg.sim, "dt", 0.01)) * float(
            getattr(self.env_cfg, "decimation", 1)
        )
        self._yaw_est = 0.0
        self._trail_xy = []
        self._trail_xz = []
        self._fallback_state = {"pseudo_pos": np.zeros(3, dtype=float)}

        if self.save_trajectory and trajectory_output_file is None:
            # Default sibling next to the MP4: foo.mp4 -> foo.pt
            base, _ = os.path.splitext(output_file)
            trajectory_output_file = base + ".pt"
        self._traj_reset(self.env.unwrapped, checkpoint_path=checkpoint)

        print(f"[INFO] Device: {self.device}")
        print(f"[INFO] Task: {self.task}")
        print(f"[INFO] Output video: {output_file}")
        print(f"[INFO] Num envs: {self.env_cfg.scene.num_envs}")
        if self.save_trajectory:
            print(f"[INFO] Output trajectory: {trajectory_output_file}")

        saved_traj_path: str | None = None
        try:
            if self.rl_library == "rsl_rl":
                self._record_rsl_rl(self.env, self.device, self.simulation_app, writer, active_env_idx, load_cfg_from_registry, torch, checkpoint)
            elif self.rl_library == "rl_games":
                self._record_rl_games(self.env, self.device, self.simulation_app, writer, active_env_idx, load_cfg_from_registry, torch, checkpoint)
            else:
                raise ValueError(f"Unsupported rl_library: {self.rl_library}")
        finally:
            writer.close()
            print(f"[INFO] Fallback recording complete. Frames: {self._frame_idx}")
            print(f"[INFO] Saved video: {output_file}")
            if self.save_trajectory and trajectory_output_file is not None:
                try:
                    saved_traj_path = self._traj_save(trajectory_output_file)
                    if saved_traj_path:
                        print(
                            f"[INFO] Saved trajectory: {saved_traj_path} "
                            f"({self._traj_n_episodes_completed} eps, "
                            f"{len(self._traj_step_action)} env-steps)"
                        )
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] Failed to save trajectory file {trajectory_output_file!r}: {exc}")
        return saved_traj_path

    def close(self):
        """Close the environment and simulation app."""
        self.env.close()
        self.simulation_app.close()

    def _traj_reset(self, base_env, checkpoint_path: str | None) -> None:
        """Clear trajectory buffers at the start of a new ``record()`` call."""
        if not self.save_trajectory:
            self._traj_active = False
            return

        import torch  # local import: torch may not be importable until AppLauncher ran

        n_envs = int(getattr(base_env, "num_envs", self.num_envs) or 1)
        self._traj_step_obs = []
        self._traj_step_action = []
        self._traj_step_oracle = []
        self._traj_step_done = []
        self._traj_step_episode_id = []
        self._traj_step_state = {}
        self._traj_episode_id = torch.zeros(n_envs, dtype=torch.long)
        self._traj_episode_oracle_returns = []
        self._traj_oracle_return_buf = torch.zeros(n_envs, dtype=torch.float32)
        self._traj_discount_buf = torch.ones(n_envs, dtype=torch.float32)
        self._traj_n_episodes_completed = 0
        self._traj_checkpoint_path = checkpoint_path
        self._traj_active = True

    def _traj_record_step(
        self,
        base_env,
        obs_tensor,
        actions,
        rewards,
        done_mask,
    ) -> None:
        """Append one env-step to the trajectory buffers.

        ``rewards`` is the per-env oracle-reward tensor returned by ``env.step``;
        we discount it with ``self.trajectory_gamma`` for the per-episode return
        summary, but the saved per-step tensor is verbatim so any other gamma can
        be applied later by the offline-replay tooling.
        """
        if not (self.save_trajectory and self._traj_active):
            return

        import torch

        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)
        rewards_cpu = rewards.detach().to(torch.float32).cpu().reshape(-1)
        done_cpu = done_mask.detach().bool().cpu().reshape(-1)

        # Allow the buffer to lazily resize if e.g. num_envs disagrees with init.
        n_envs = rewards_cpu.shape[0]
        if self._traj_oracle_return_buf is None or self._traj_oracle_return_buf.shape[0] != n_envs:
            self._traj_oracle_return_buf = torch.zeros(n_envs, dtype=torch.float32)
            self._traj_discount_buf = torch.ones(n_envs, dtype=torch.float32)
            self._traj_episode_id = torch.zeros(n_envs, dtype=torch.long)

        self._traj_step_obs.append(obs_tensor.detach().cpu().clone())
        self._traj_step_action.append(actions.detach().cpu().clone())
        self._traj_step_oracle.append(rewards_cpu)
        self._traj_step_done.append(done_cpu)
        self._traj_step_episode_id.append(self._traj_episode_id.clone())

        for name, tensor in _capture_state(base_env).items():
            self._traj_step_state.setdefault(name, []).append(tensor)

        self._traj_oracle_return_buf += self._traj_discount_buf * rewards_cpu
        self._traj_discount_buf = self._traj_discount_buf * self.trajectory_gamma

        if done_cpu.any():
            done_idx = torch.nonzero(done_cpu, as_tuple=False).flatten().tolist()
            for i in done_idx:
                self._traj_episode_oracle_returns.append(
                    float(self._traj_oracle_return_buf[i].item())
                )
                self._traj_n_episodes_completed += 1
            self._traj_oracle_return_buf[done_cpu] = 0.0
            self._traj_discount_buf[done_cpu] = 1.0
            self._traj_episode_id[done_cpu] += 1

    def _traj_save(self, output_path: str) -> str | None:
        """Stack buffers and write a ``.pt`` file in the offline-replay schema.

        Returns the absolute path written, or ``None`` if no steps were recorded
        (e.g. ``save_trajectory=False`` or the loop exited before the first step).
        """
        if not (self.save_trajectory and self._traj_active and self._traj_step_action):
            self._traj_active = False
            return None

        import torch

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        traj: dict = {
            "obs": torch.stack(self._traj_step_obs, dim=0),
            "action": torch.stack(self._traj_step_action, dim=0),
            "oracle_reward": torch.stack(self._traj_step_oracle, dim=0),
            # No candidate reward installed in the recorder env (only the policy is
            # loaded, not the LLM reward). The offline replay tooling synthesises
            # the candidate reward from `state` + `action`. We still emit a zero
            # tensor of the right shape so downstream schema checks pass.
            "candidate_reward": torch.zeros_like(torch.stack(self._traj_step_oracle, dim=0)),
            "done": torch.stack(self._traj_step_done, dim=0),
            "episode_id": torch.stack(self._traj_step_episode_id, dim=0),
            "episode_oracle_returns": torch.tensor(self._traj_episode_oracle_returns),
            "episode_candidate_returns": torch.zeros(
                len(self._traj_episode_oracle_returns), dtype=torch.float32
            ),
            "n_episodes_completed": int(self._traj_n_episodes_completed),
            "n_env_steps": int(len(self._traj_step_action)),
            "gamma": float(self.trajectory_gamma),
            "task": self.task,
            "checkpoint_path": self._traj_checkpoint_path,
        }
        if self._traj_step_state:
            traj["state"] = {
                name: torch.stack(seq, dim=0) for name, seq in self._traj_step_state.items()
            }
        torch.save(traj, output_path)
        self._traj_active = False
        return os.path.abspath(output_path)

    def _record_rsl_rl(self, env, device, simulation_app, writer, active_env_idx, load_cfg_from_registry, torch, checkpoint):
        """Record using RSL-RL library."""
        from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner

        agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(self.task, "rsl_rl_cfg_entry_point")
        agent_cfg.device = device

        env = RslRlVecEnvWrapper(env)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        obs = env.get_observations()

        while simulation_app.is_running():
            with torch.inference_mode():
                actions = policy(obs)
            obs, rewards, dones, _ = env.step(actions)

            policy_obs = self._extract_policy_obs(obs)
            if policy_obs.ndim == 1:
                policy_obs = policy_obs.unsqueeze(0)
            env_idx = min(active_env_idx, policy_obs.shape[0] - 1)
            obs_row = policy_obs[env_idx].detach().cpu().numpy()

            if obs_row.shape[0] < 12:
                raise RuntimeError(f"Expected Quadcopter obs with >=12 values. Got shape: {obs_row.shape}")

            pos_w, desired_w, lin_vel_b, ang_vel_b = self._read_quadcopter_state(
                env.unwrapped, env_idx, obs_row
            )
            reward = float(rewards[env_idx].item())
            done = bool(dones[env_idx].item())
            distance_to_goal = float(np.linalg.norm(desired_w - pos_w))
            self._yaw_est += float(ang_vel_b[2]) * self._step_dt

            self._update_trails(pos_w)

            self._traj_record_step(
                base_env=env.unwrapped,
                obs_tensor=policy_obs,
                actions=actions,
                rewards=rewards,
                done_mask=dones,
            )

            frame = self._render_frame(
                pos_w=pos_w,
                desired_w=desired_w,
                lin_vel_b=lin_vel_b,
                ang_vel_b=ang_vel_b,
                reward=reward,
                done=done,
                distance_to_goal=distance_to_goal,
            )
            writer.append_data(frame)

            self._frame_idx += 1
            self._episode_step += 1
            if done:
                self._episode_idx += 1
                self._episode_step = 0

            if self._frame_idx % 100 == 0:
                print(f"[INFO] Recorded {self._frame_idx} frames (episodes finished: {self._episode_idx})")

            if self._should_stop():
                break

    def _record_rl_games(self, env, device, simulation_app, writer, active_env_idx, load_cfg_from_registry, torch, checkpoint):
        """Record using RL-Games library."""
        from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
        from rl_games.common import env_configurations, vecenv
        from rl_games.common.algo_observer import IsaacAlgoObserver
        from rl_games.torch_runner import Runner

        agent_cfg = load_cfg_from_registry(self.task, "rl_games_cfg_entry_point")
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = checkpoint
        agent_cfg["params"]["config"]["device"] = device
        agent_cfg["params"]["config"]["device_name"] = device
        clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
        clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
        env = RlGamesVecEnvWrapper(env, device, clip_obs, clip_actions)

        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
        )
        env_configurations.register(
            "rlgpu",
            {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env},
        )

        agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
        runner = Runner(IsaacAlgoObserver())
        runner.load(agent_cfg)

        from rl_games.common.player import BasePlayer

        agent: BasePlayer = runner.create_player()
        agent.restore(checkpoint)
        agent.reset()

        obs = env.reset()
        if isinstance(obs, dict) and "obs" in obs:
            obs = obs["obs"]
        _ = agent.get_batch_size(obs, 1)
        if agent.is_rnn:
            agent.init_rnn()

        while simulation_app.is_running():
            with torch.inference_mode():
                actor_obs = agent.obs_to_torch(obs)
                actions = agent.get_action(actor_obs, is_deterministic=True)
            next_obs, rewards, dones, _ = env.step(actions)

            policy_obs = self._extract_policy_obs(next_obs)
            if policy_obs.ndim == 1:
                policy_obs = policy_obs.unsqueeze(0)
            env_idx = min(active_env_idx, policy_obs.shape[0] - 1)
            obs_row = policy_obs[env_idx].detach().cpu().numpy()

            if obs_row.shape[0] < 12:
                raise RuntimeError(f"Expected Quadcopter obs with >=12 values. Got shape: {obs_row.shape}")

            pos_w, desired_w, lin_vel_b, ang_vel_b = self._read_quadcopter_state(
                env.unwrapped, env_idx, obs_row
            )
            reward = float(rewards[env_idx].item())
            done = bool(dones[env_idx].item())
            distance_to_goal = float(np.linalg.norm(desired_w - pos_w))
            self._yaw_est += float(ang_vel_b[2]) * self._step_dt

            self._update_trails(pos_w)

            self._traj_record_step(
                base_env=env.unwrapped,
                obs_tensor=policy_obs,
                actions=actions,
                rewards=rewards,
                done_mask=dones,
            )

            frame = self._render_frame(
                pos_w=pos_w,
                desired_w=desired_w,
                lin_vel_b=lin_vel_b,
                ang_vel_b=ang_vel_b,
                reward=reward,
                done=done,
                distance_to_goal=distance_to_goal,
            )
            writer.append_data(frame)

            self._frame_idx += 1
            self._episode_step += 1
            if done:
                self._episode_idx += 1
                self._episode_step = 0

            if self._frame_idx % 100 == 0:
                print(f"[INFO] Recorded {self._frame_idx} frames (episodes finished: {self._episode_idx})")

            if len(dones) > 0 and agent.is_rnn and agent.states is not None:
                for state_buf in agent.states:
                    state_buf[:, dones, :] = 0.0

            obs = next_obs["obs"] if isinstance(next_obs, dict) and "obs" in next_obs else next_obs

            if self._should_stop():
                break
