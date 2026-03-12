# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

"""Record Quadcopter policy rollouts without Isaac camera rendering.

This script runs policy inference headlessly and generates a lightweight 2D MP4
visualization from simulation states and observations. It is intended for
clusters where Isaac renderer-based recording is unavailable.
"""

from __future__ import annotations

import argparse
import math
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from isaaclab_eureka.utils import get_freest_gpu


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


def _project_to_panel(x: float, y: float, bounds: tuple[float, float, float, float], panel: tuple[int, int, int, int]):
    x_min, x_max, y_min, y_max = bounds
    px0, py0, px1, py1 = panel
    span_x = max(1e-6, x_max - x_min)
    span_y = max(1e-6, y_max - y_min)
    px = px0 + (x - x_min) / span_x * (px1 - px0)
    py = py1 - (y - y_min) / span_y * (py1 - py0)
    return int(px), int(py)


def _compute_bounds_2d(
    points: list[tuple[float, float]],
    min_span: float = 4.0,
    margin_ratio: float = 0.15,
) -> tuple[float, float, float, float]:
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
    draw: ImageDraw.ImageDraw,
    trail: list[tuple[float, float]],
    bounds: tuple[float, float, float, float],
    panel: tuple[int, int, int, int],
    color: tuple[int, int, int],
):
    if len(trail) < 2:
        return
    points = [_project_to_panel(x, y, bounds, panel) for (x, y) in trail]
    draw.line(points, fill=color, width=3)


def _draw_drone_topdown(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    yaw: float,
    bounds: tuple[float, float, float, float],
    panel: tuple[int, int, int, int],
):
    cx, cy = _project_to_panel(x, y, bounds, panel)
    r = 8
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(38, 132, 255), outline=(10, 60, 120), width=2)

    arm = 18
    fx = cx + int(arm * math.cos(yaw))
    fy = cy - int(arm * math.sin(yaw))
    draw.line([(cx, cy), (fx, fy)], fill=(10, 60, 120), width=3)


def _draw_goal(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    bounds: tuple[float, float, float, float],
    panel: tuple[int, int, int, int],
):
    gx, gy = _project_to_panel(x, y, bounds, panel)
    s = 8
    draw.line([(gx - s, gy), (gx + s, gy)], fill=(204, 53, 53), width=3)
    draw.line([(gx, gy - s), (gx, gy + s)], fill=(204, 53, 53), width=3)
    draw.ellipse([gx - s - 3, gy - s - 3, gx + s + 3, gy + s + 3], outline=(204, 53, 53), width=2)


def _render_quadcopter_frame(
    pos_w: np.ndarray,
    desired_w: np.ndarray,
    lin_vel_b: np.ndarray,
    ang_vel_b: np.ndarray,
    reward: float,
    done: bool,
    frame_idx: int,
    episode_idx: int,
    episode_step: int,
    distance_to_goal: float,
    yaw_est: float,
    trail_xy: list[tuple[float, float]],
    trail_xz: list[tuple[float, float]],
    width: int,
    height: int,
) -> np.ndarray:
    image = Image.new("RGB", (width, height), (246, 247, 250))
    draw = ImageDraw.Draw(image)

    left_panel = (30, 70, width // 2 - 20, height - 40)
    right_panel = (width // 2 + 20, 70, width - 30, height - 40)

    draw.rectangle(left_panel, outline=(86, 94, 106), width=2)
    draw.rectangle(right_panel, outline=(86, 94, 106), width=2)
    draw.text((left_panel[0], 30), "Top-down (x-y)", fill=(30, 30, 30))
    draw.text((right_panel[0], 30), "Side view (x-z)", fill=(30, 30, 30))

    xy_bounds = _compute_bounds_2d(
        [(pos_w[0], pos_w[1]), (desired_w[0], desired_w[1])]
        + trail_xy[-30:],
        min_span=4.0,
        margin_ratio=0.2,
    )
    z_bounds = _compute_bounds_2d(
        [(pos_w[0], pos_w[2]), (desired_w[0], desired_w[2])]
        + trail_xz[-30:],
        min_span=2.5,
        margin_ratio=0.2,
    )

    _draw_trail(draw, trail_xy, xy_bounds, left_panel, (102, 157, 246))
    _draw_goal(draw, desired_w[0], desired_w[1], xy_bounds, left_panel)
    _draw_drone_topdown(draw, pos_w[0], pos_w[1], yaw_est, xy_bounds, left_panel)

    _draw_trail(draw, trail_xz, z_bounds, right_panel, (102, 157, 246))
    _draw_goal(draw, desired_w[0], desired_w[2], z_bounds, right_panel)
    dx, dz = _project_to_panel(pos_w[0], pos_w[2], z_bounds, right_panel)
    r = 8
    draw.ellipse([dx - r, dz - r, dx + r, dz + r], fill=(38, 132, 255), outline=(10, 60, 120), width=2)

    status = "DONE" if done else "RUN"
    status_color = (173, 25, 25) if done else (32, 32, 32)
    text_lines = [
        f"status={status} frame={frame_idx} episode={episode_idx} step={episode_step}",
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


def _read_quadcopter_state(base_env, env_index: int, obs_row: np.ndarray, step_dt: float, fallback_state: dict):
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
    desired_pos_b = np.array(obs_row[9:12], dtype=float) if obs_row.shape[0] >= 12 else np.zeros(3, dtype=float)

    if pos_w is None:
        fallback_state["pseudo_pos"] = fallback_state["pseudo_pos"] + lin_vel_b * step_dt
        pos_w = fallback_state["pseudo_pos"].copy()
    if desired_w is None:
        desired_w = pos_w + desired_pos_b

    return pos_w, desired_w, lin_vel_b, ang_vel_b


def main(args_cli):
    from isaaclab.app import AppLauncher

    device = args_cli.device
    if device == "cuda":
        device_id = get_freest_gpu()
        device = f"cuda:{device_id}"

    app_launcher = AppLauncher(headless=args_cli.headless, device=device)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import torch
    from isaaclab.envs import DirectRLEnvCfg
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    env_cfg: DirectRLEnvCfg = parse_env_cfg(args_cli.task)
    env_cfg.sim.device = device
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1
    env = gym.make(args_cli.task, cfg=env_cfg)

    os.makedirs(os.path.dirname(args_cli.output_file) or ".", exist_ok=True)
    writer = imageio.get_writer(args_cli.output_file, fps=args_cli.fps)

    env_index = max(0, args_cli.env_index)
    frame_idx = 0
    episode_idx = 0
    episode_step = 0
    step_dt = float(getattr(env_cfg.sim, "dt", 0.01)) * float(getattr(env_cfg, "decimation", 1))
    yaw_est = 0.0
    trail_xy: list[tuple[float, float]] = []
    trail_xz: list[tuple[float, float]] = []
    fallback_state = {"pseudo_pos": np.zeros(3, dtype=float)}

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Checkpoint: {args_cli.checkpoint}")
    print(f"[INFO] Output video: {args_cli.output_file}")
    print(f"[INFO] Num envs: {env_cfg.scene.num_envs}")

    try:
        if args_cli.rl_library == "rsl_rl":
            from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
            from rsl_rl.runners import OnPolicyRunner

            agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
            agent_cfg.device = device

            env = RslRlVecEnvWrapper(env)
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(args_cli.checkpoint)
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            obs = env.get_observations()

            while simulation_app.is_running():
                with torch.inference_mode():
                    actions = policy(obs)
                    obs, rewards, dones, _ = env.step(actions)

                policy_obs = _extract_policy_obs(obs)
                if policy_obs.ndim == 1:
                    policy_obs = policy_obs.unsqueeze(0)
                active_env_idx = min(env_index, policy_obs.shape[0] - 1)
                obs_row = policy_obs[active_env_idx].detach().cpu().numpy()

                if obs_row.shape[0] < 12:
                    raise RuntimeError(f"Expected Quadcopter obs with >=12 values. Got shape: {obs_row.shape}")

                pos_w, desired_w, lin_vel_b, ang_vel_b = _read_quadcopter_state(
                    env.unwrapped, active_env_idx, obs_row, step_dt, fallback_state
                )
                reward = float(rewards[active_env_idx].item())
                done = bool(dones[active_env_idx].item())
                distance_to_goal = float(np.linalg.norm(desired_w - pos_w))
                yaw_est += float(ang_vel_b[2]) * step_dt

                trail_xy.append((float(pos_w[0]), float(pos_w[1])))
                trail_xz.append((float(pos_w[0]), float(pos_w[2])))
                if len(trail_xy) > args_cli.trail_length:
                    trail_xy = trail_xy[-args_cli.trail_length :]
                    trail_xz = trail_xz[-args_cli.trail_length :]

                frame = _render_quadcopter_frame(
                    pos_w=pos_w,
                    desired_w=desired_w,
                    lin_vel_b=lin_vel_b,
                    ang_vel_b=ang_vel_b,
                    reward=reward,
                    done=done,
                    frame_idx=frame_idx,
                    episode_idx=episode_idx,
                    episode_step=episode_step,
                    distance_to_goal=distance_to_goal,
                    yaw_est=yaw_est,
                    trail_xy=trail_xy,
                    trail_xz=trail_xz,
                    width=args_cli.frame_width,
                    height=args_cli.frame_height,
                )
                writer.append_data(frame)

                frame_idx += 1
                episode_step += 1
                if done:
                    episode_idx += 1
                    episode_step = 0

                if frame_idx % 100 == 0:
                    print(f"[INFO] Recorded {frame_idx} frames (episodes finished: {episode_idx})")

                if args_cli.max_frames > 0 and frame_idx >= args_cli.max_frames:
                    break
                if args_cli.num_episodes > 0 and episode_idx >= args_cli.num_episodes:
                    break

        elif args_cli.rl_library == "rl_games":
            from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
            from rl_games.common import env_configurations, vecenv
            from rl_games.common.algo_observer import IsaacAlgoObserver
            from rl_games.torch_runner import Runner

            agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")
            agent_cfg["params"]["load_checkpoint"] = True
            agent_cfg["params"]["load_path"] = args_cli.checkpoint
            agent_cfg["params"]["config"]["device"] = device
            agent_cfg["params"]["config"]["device_name"] = device
            clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
            clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
            env = RlGamesVecEnvWrapper(env, device, clip_obs, clip_actions)

            vecenv.register(
                "IsaacRlgWrapper",
                lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
            )
            env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

            agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
            runner = Runner(IsaacAlgoObserver())
            runner.load(agent_cfg)

            from rl_games.common.player import BasePlayer

            agent: BasePlayer = runner.create_player()
            agent.restore(args_cli.checkpoint)
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

                policy_obs = _extract_policy_obs(next_obs)
                if policy_obs.ndim == 1:
                    policy_obs = policy_obs.unsqueeze(0)
                active_env_idx = min(env_index, policy_obs.shape[0] - 1)
                obs_row = policy_obs[active_env_idx].detach().cpu().numpy()

                if obs_row.shape[0] < 12:
                    raise RuntimeError(f"Expected Quadcopter obs with >=12 values. Got shape: {obs_row.shape}")

                pos_w, desired_w, lin_vel_b, ang_vel_b = _read_quadcopter_state(
                    env.unwrapped, active_env_idx, obs_row, step_dt, fallback_state
                )
                reward = float(rewards[active_env_idx].item())
                done = bool(dones[active_env_idx].item())
                distance_to_goal = float(np.linalg.norm(desired_w - pos_w))
                yaw_est += float(ang_vel_b[2]) * step_dt

                trail_xy.append((float(pos_w[0]), float(pos_w[1])))
                trail_xz.append((float(pos_w[0]), float(pos_w[2])))
                if len(trail_xy) > args_cli.trail_length:
                    trail_xy = trail_xy[-args_cli.trail_length :]
                    trail_xz = trail_xz[-args_cli.trail_length :]

                frame = _render_quadcopter_frame(
                    pos_w=pos_w,
                    desired_w=desired_w,
                    lin_vel_b=lin_vel_b,
                    ang_vel_b=ang_vel_b,
                    reward=reward,
                    done=done,
                    frame_idx=frame_idx,
                    episode_idx=episode_idx,
                    episode_step=episode_step,
                    distance_to_goal=distance_to_goal,
                    yaw_est=yaw_est,
                    trail_xy=trail_xy,
                    trail_xz=trail_xz,
                    width=args_cli.frame_width,
                    height=args_cli.frame_height,
                )
                writer.append_data(frame)

                frame_idx += 1
                episode_step += 1
                if done:
                    episode_idx += 1
                    episode_step = 0

                if frame_idx % 100 == 0:
                    print(f"[INFO] Recorded {frame_idx} frames (episodes finished: {episode_idx})")

                if len(dones) > 0 and agent.is_rnn and agent.states is not None:
                    for state_buf in agent.states:
                        state_buf[:, dones, :] = 0.0

                obs = next_obs["obs"] if isinstance(next_obs, dict) and "obs" in next_obs else next_obs

                if args_cli.max_frames > 0 and frame_idx >= args_cli.max_frames:
                    break
                if args_cli.num_episodes > 0 and episode_idx >= args_cli.num_episodes:
                    break

        else:
            raise ValueError(f"Unsupported rl_library: {args_cli.rl_library}")

    finally:
        writer.close()
        env.close()
        simulation_app.close()
        print(f"[INFO] Fallback recording complete. Frames: {frame_idx}")
        print(f"[INFO] Saved video: {args_cli.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record Quadcopter rollout via observation/state fallback rendering.")
    parser.add_argument("--task", type=str, default="Isaac-Quadcopter-Direct-v0", help="Name of the task.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for simulation and policy inference.")
    parser.add_argument(
        "--rl_library",
        type=str,
        default="rsl_rl",
        choices=["rsl_rl", "rl_games"],
        help="The RL library used to train the checkpoint.",
    )
    parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
    parser.add_argument("--output_file", type=str, default="./recordings/quadcopter_fallback.mp4", help="Output MP4 path.")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS.")
    parser.add_argument("--max_frames", type=int, default=900, help="Maximum frames to write. <=0 means unlimited.")
    parser.add_argument("--num_episodes", type=int, default=1, help="Stop after this many completed episodes. <=0 means unlimited.")
    parser.add_argument("--env_index", type=int, default=0, help="Environment index to visualize.")
    parser.add_argument("--trail_length", type=int, default=160, help="How many recent positions to draw as trajectory.")
    parser.add_argument("--frame_width", type=int, default=1280, help="Video frame width in pixels.")
    parser.add_argument("--frame_height", type=int, default=720, help="Video frame height in pixels.")
    args_cli = parser.parse_args()
    main(args_cli)
