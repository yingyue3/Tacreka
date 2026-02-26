# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

"""Record Cartpole policy rollouts without Isaac camera rendering.

This script is intended for environments where Vulkan/display-based recording is unavailable.
It runs policy inference headlessly, reads Cartpole state from observations, and draws a simple
2D animation into an MP4 file.
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


def _render_cartpole_frame(
    cart_pos: float,
    pole_angle: float,
    cart_vel: float,
    pole_vel: float,
    reward: float,
    episode_idx: int,
    episode_step: int,
    frame_idx: int,
    done: bool,
    max_cart_pos: float,
    width: int,
    height: int,
) -> np.ndarray:
    """Draw a lightweight cartpole frame from state values."""
    image = Image.new("RGB", (width, height), (246, 247, 250))
    draw = ImageDraw.Draw(image)

    rail_y = int(height * 0.68)
    margin_x = int(width * 0.08)
    rail_left = margin_x
    rail_right = width - margin_x
    draw.line([(rail_left, rail_y), (rail_right, rail_y)], fill=(80, 88, 97), width=6)

    safe_max = max(1e-6, float(max_cart_pos))
    clamped_cart = max(-safe_max, min(safe_max, cart_pos))
    normalized = clamped_cart / safe_max
    mid_x = (rail_left + rail_right) / 2.0
    span = (rail_right - rail_left) * 0.45
    cart_center_x = int(mid_x + normalized * span)
    cart_center_y = rail_y - int(height * 0.08)

    cart_w = int(width * 0.09)
    cart_h = int(height * 0.08)
    cart_bbox = [
        cart_center_x - cart_w // 2,
        cart_center_y - cart_h // 2,
        cart_center_x + cart_w // 2,
        cart_center_y + cart_h // 2,
    ]
    draw.rounded_rectangle(cart_bbox, radius=10, fill=(42, 99, 204), outline=(27, 59, 111), width=3)

    wheel_r = max(4, int(height * 0.015))
    wheel_y = cart_bbox[3] + wheel_r + 2
    draw.ellipse(
        [cart_bbox[0] + wheel_r, wheel_y - wheel_r, cart_bbox[0] + 3 * wheel_r, wheel_y + wheel_r],
        fill=(45, 45, 45),
    )
    draw.ellipse(
        [cart_bbox[2] - 3 * wheel_r, wheel_y - wheel_r, cart_bbox[2] - wheel_r, wheel_y + wheel_r],
        fill=(45, 45, 45),
    )

    pole_base = (cart_center_x, cart_bbox[1] + int(height * 0.01))
    pole_len = int(height * 0.30)
    # Cartpole task angle is 0 when upright.
    pole_tip = (
        int(pole_base[0] + pole_len * math.sin(pole_angle)),
        int(pole_base[1] - pole_len * math.cos(pole_angle)),
    )
    draw.line([pole_base, pole_tip], fill=(214, 78, 36), width=max(4, int(height * 0.014)))
    joint_r = max(4, int(height * 0.012))
    draw.ellipse(
        [pole_base[0] - joint_r, pole_base[1] - joint_r, pole_base[0] + joint_r, pole_base[1] + joint_r],
        fill=(255, 183, 77),
    )

    status = "DONE" if done else "RUN"
    txt_color = (183, 28, 28) if done else (32, 32, 32)
    info_lines = [
        f"status={status}  frame={frame_idx}",
        f"episode={episode_idx}  episode_step={episode_step}",
        f"cart_pos={cart_pos:+.3f}  cart_vel={cart_vel:+.3f}",
        f"pole_angle={pole_angle:+.3f}  pole_vel={pole_vel:+.3f}",
        f"reward={reward:+.3f}",
    ]
    y0 = int(height * 0.05)
    for line in info_lines:
        draw.text((margin_x, y0), line, fill=txt_color)
        y0 += int(height * 0.045)

    return np.asarray(image, dtype=np.uint8)


def main(args_cli):
    """Run a checkpoint policy and create a fallback MP4 recording."""
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
    episode_idx = 0
    episode_step = 0
    frame_idx = 0

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Checkpoint: {args_cli.checkpoint}")
    print(f"[INFO] Output video: {args_cli.output_file}")
    print(f"[INFO] Number of environments: {env_cfg.scene.num_envs}")

    try:
        if args_cli.rl_library == "rsl_rl":
            from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
            from rsl_rl.runners import OnPolicyRunner

            agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
            agent_cfg.device = device

            env = RslRlVecEnvWrapper(env)
            ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            ppo_runner.load(args_cli.checkpoint)
            policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
            obs = env.get_observations()

            while simulation_app.is_running():
                with torch.inference_mode():
                    actions = policy(obs)
                    obs, rewards, dones, _ = env.step(actions)

                policy_obs = _extract_policy_obs(obs)
                if policy_obs.ndim == 1:
                    policy_obs = policy_obs.unsqueeze(0)
                active_env_idx = min(env_index, policy_obs.shape[0] - 1)
                state = policy_obs[active_env_idx].detach().cpu()

                if state.numel() < 4:
                    raise RuntimeError(f"Expected Cartpole obs with at least 4 values. Got shape: {tuple(state.shape)}")

                # Cartpole direct observation ordering:
                # [pole_angle, pole_velocity, cart_position, cart_velocity]
                pole_angle = float(state[0].item())
                pole_vel = float(state[1].item())
                cart_pos = float(state[2].item())
                cart_vel = float(state[3].item())
                reward = float(rewards[active_env_idx].item())
                done = bool(dones[active_env_idx].item())

                frame = _render_cartpole_frame(
                    cart_pos=cart_pos,
                    pole_angle=pole_angle,
                    cart_vel=cart_vel,
                    pole_vel=pole_vel,
                    reward=reward,
                    episode_idx=episode_idx,
                    episode_step=episode_step,
                    frame_idx=frame_idx,
                    done=done,
                    max_cart_pos=float(getattr(env_cfg, "max_cart_pos", 3.0)),
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
                state = policy_obs[active_env_idx].detach().cpu()

                if state.numel() < 4:
                    raise RuntimeError(f"Expected Cartpole obs with at least 4 values. Got shape: {tuple(state.shape)}")

                pole_angle = float(state[0].item())
                pole_vel = float(state[1].item())
                cart_pos = float(state[2].item())
                cart_vel = float(state[3].item())
                reward = float(rewards[active_env_idx].item())
                done = bool(dones[active_env_idx].item())

                frame = _render_cartpole_frame(
                    cart_pos=cart_pos,
                    pole_angle=pole_angle,
                    cart_vel=cart_vel,
                    pole_vel=pole_vel,
                    reward=reward,
                    episode_idx=episode_idx,
                    episode_step=episode_step,
                    frame_idx=frame_idx,
                    done=done,
                    max_cart_pos=float(getattr(env_cfg, "max_cart_pos", 3.0)),
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
    parser = argparse.ArgumentParser(description="Record Cartpole rollout via observation-based fallback rendering.")
    parser.add_argument("--task", type=str, default="Isaac-Cartpole-Direct-v0", help="Name of the task.")
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
    parser.add_argument("--output_file", type=str, default="./recordings/cartpole_fallback.mp4", help="Output MP4 path.")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS.")
    parser.add_argument("--max_frames", type=int, default=600, help="Maximum frames to write. <=0 means unlimited.")
    parser.add_argument("--num_episodes", type=int, default=1, help="Stop after this many completed episodes. <=0 means unlimited.")
    parser.add_argument("--env_index", type=int, default=0, help="Environment index to visualize.")
    parser.add_argument("--frame_width", type=int, default=960, help="Video frame width in pixels.")
    parser.add_argument("--frame_height", type=int, default=540, help="Video frame height in pixels.")
    args_cli = parser.parse_args()
    main(args_cli)
