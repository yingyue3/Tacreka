# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

"""Record Isaac-Lift-Cube-Franka-v0 policy rollouts without Isaac camera rendering.

Runs policy inference headlessly and produces a lightweight 2D MP4 that shows
the Franka arm's end-effector, the cube, and the commanded goal position from
two orthogonal viewpoints (side x-z and top-down x-y).  Intended for clusters
where Isaac renderer-based recording is unavailable.

Observation layout (37 dims, from FrankaCubeLiftEnvCfg.ObservationsCfg.PolicyCfg):
  obs[ 0: 9]   joint_pos_rel  (7 arm + 2 finger, relative to defaults)
  obs[ 9:18]   joint_vel_rel  (7 arm + 2 finger)
  obs[18:21]   object_position_in_robot_root_frame  (cube x, y, z in robot frame)
  obs[21:28]   generated_commands / object_pose  (pos[3] + quat[4] in robot frame)
  obs[28:37]   last_action    (9)

Scene state (read directly from ManagerBasedRLEnv at each step):
  env.scene["object"].data.root_pos_w[i]              cube world pos
  env.scene["ee_frame"].data.target_pos_w[i, 0, :]    EE  world pos
  env.scene["robot"].data.root_pos_w[i]               robot base world pos
  env.command_manager.get_command("object_pose")[i,:3] goal pos in robot frame
  goal_world = robot_base_pos + goal_robot_frame       (identity base rotation)
"""

from __future__ import annotations

import argparse
import math
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from isaaclab_eureka.utils import get_freest_gpu


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
_C_BG         = (246, 247, 250)
_C_PANEL_BG   = (252, 252, 253)
_C_BORDER     = (86,  94,  106)
_C_TEXT       = (32,  32,  32)
_C_TEXT_ERR   = (173, 25,  25)
_C_TEXT_OK    = (22,  120, 60)

_C_TABLE      = (160, 130, 90)    # brown – table surface line
_C_ROBOT_BASE = (100, 100, 110)   # dark grey – robot footprint
_C_ARM        = (180, 185, 195)   # light grey – arm stick
_C_LIFT_LINE  = (200, 200, 200)   # dashed lift-height threshold
_C_GOAL_LINE  = (200, 200, 200)   # goal height dashed

_C_EE         = (38,  132, 255)   # blue – end-effector
_C_EE_TRAIL   = (140, 190, 255)   # light blue – EE trail
_C_CUBE       = (255, 140, 30)    # orange – cube
_C_CUBE_TRAIL = (255, 210, 140)   # light orange – cube trail
_C_GOAL       = (200, 40,  40)    # red – goal marker

# ---------------------------------------------------------------------------
# Minimum bounds so that panels are never empty
# ---------------------------------------------------------------------------
_SIDE_X_SPAN  = 1.0   # metres in x (forward)
_SIDE_Z_SPAN  = 0.70  # metres in z (height)
_TOP_X_SPAN   = 0.80  # metres in x
_TOP_Y_SPAN   = 0.70  # metres in y (lateral)


# ---------------------------------------------------------------------------
# Shared geometry helpers
# ---------------------------------------------------------------------------

def _extract_policy_obs(obs):
    """Unwrap TensorDict / nested-dict obs to the 'policy' tensor."""
    if hasattr(obs, "keys"):
        keys = set(obs.keys())
        if "policy" in keys:
            return obs["policy"]
        if "obs" in keys:
            inner = obs["obs"]
            if isinstance(inner, dict):
                return inner.get("policy", next(iter(inner.values())))
            return inner
    return obs


def _project(x: float, y: float,
             bounds: tuple[float, float, float, float],
             panel: tuple[int, int, int, int]) -> tuple[int, int]:
    """Map a 2D world coordinate to a panel pixel coordinate."""
    x_min, x_max, y_min, y_max = bounds
    px0, py0, px1, py1 = panel
    span_x = max(1e-9, x_max - x_min)
    span_y = max(1e-9, y_max - y_min)
    px = px0 + (x - x_min) / span_x * (px1 - px0)
    py = py1 - (y - y_min) / span_y * (py1 - py0)
    return int(px), int(py)


def _fixed_bounds(cx: float, cy: float,
                  half_x: float, half_y: float,
                  margin: float = 0.12) -> tuple[float, float, float, float]:
    """Return axis-aligned bounds centred at (cx, cy) with fixed half-extents."""
    hx = half_x * (1.0 + margin)
    hy = half_y * (1.0 + margin)
    return cx - hx, cx + hx, cy - hy, cy + hy


def _adaptive_bounds(points: list[tuple[float, float]],
                     min_x_span: float, min_y_span: float,
                     margin: float = 0.15) -> tuple[float, float, float, float]:
    """Return bounds that contain all points, with minimum spans and a margin."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    cx = 0.5 * (min(xs) + max(xs))
    cy = 0.5 * (min(ys) + max(ys))
    hx = 0.5 * max(min_x_span, max(xs) - min(xs)) * (1.0 + margin)
    hy = 0.5 * max(min_y_span, max(ys) - min(ys)) * (1.0 + margin)
    return cx - hx, cx + hx, cy - hy, cy + hy


def _draw_trail(draw: ImageDraw.ImageDraw,
                trail: list[tuple[float, float]],
                bounds: tuple[float, float, float, float],
                panel: tuple[int, int, int, int],
                color: tuple[int, int, int],
                width: int = 2) -> None:
    if len(trail) < 2:
        return
    pts = [_project(x, y, bounds, panel) for x, y in trail]
    draw.line(pts, fill=color, width=width)


def _draw_dashed_hline(draw: ImageDraw.ImageDraw,
                       y_world: float,
                       bounds: tuple[float, float, float, float],
                       panel: tuple[int, int, int, int],
                       color: tuple[int, int, int],
                       dash: int = 10, gap: int = 6) -> None:
    """Draw a horizontal dashed line at world y-coordinate y_world."""
    x_min, x_max = bounds[0], bounds[1]
    px0, _, px1, _ = panel
    _, py = _project(x_min, y_world, bounds, panel)
    x = px0
    while x < px1:
        draw.line([(x, py), (min(x + dash, px1), py)], fill=color, width=1)
        x += dash + gap


def _draw_goal_marker(draw: ImageDraw.ImageDraw,
                      x: float, y: float,
                      bounds: tuple[float, float, float, float],
                      panel: tuple[int, int, int, int],
                      size: int = 10) -> None:
    """Draw a crosshair + circle at the goal position."""
    gx, gy = _project(x, y, bounds, panel)
    draw.line([(gx - size, gy), (gx + size, gy)], fill=_C_GOAL, width=2)
    draw.line([(gx, gy - size), (gx, gy + size)], fill=_C_GOAL, width=2)
    draw.ellipse([gx - size - 2, gy - size - 2,
                  gx + size + 2, gy + size + 2], outline=_C_GOAL, width=1)


def _draw_cube(draw: ImageDraw.ImageDraw,
               x: float, y: float,
               bounds: tuple[float, float, float, float],
               panel: tuple[int, int, int, int],
               lifted: bool,
               size: int = 9) -> None:
    """Draw the cube as a filled square; brighter when lifted."""
    cx, cy = _project(x, y, bounds, panel)
    fill = (255, 160, 50) if lifted else (200, 120, 40)
    draw.rectangle([cx - size, cy - size, cx + size, cy + size],
                   fill=fill, outline=(140, 80, 20), width=2)


def _draw_ee(draw: ImageDraw.ImageDraw,
             x: float, y: float,
             bounds: tuple[float, float, float, float],
             panel: tuple[int, int, int, int],
             radius: int = 8) -> None:
    """Draw the end-effector as a filled circle."""
    ex, ey = _project(x, y, bounds, panel)
    draw.ellipse([ex - radius, ey - radius, ex + radius, ey + radius],
                 fill=_C_EE, outline=(10, 60, 120), width=2)


def _draw_arm_stick(draw: ImageDraw.ImageDraw,
                    base_x: float, base_z: float,
                    ee_x: float, ee_z: float,
                    bounds: tuple[float, float, float, float],
                    panel: tuple[int, int, int, int]) -> None:
    """Draw a simple stick from robot base to end-effector."""
    bx, bz = _project(base_x, base_z, bounds, panel)
    ex, ez = _project(ee_x, ee_z, bounds, panel)
    draw.line([(bx, bz), (ex, ez)], fill=_C_ARM, width=3)
    # Draw robot base as a small rectangle
    w, h = 14, 10
    draw.rectangle([bx - w // 2, bz - h // 2, bx + w // 2, bz + h // 2],
                   fill=_C_ROBOT_BASE, outline=(60, 60, 70), width=1)


# ---------------------------------------------------------------------------
# State reading from ManagerBasedRLEnv
# ---------------------------------------------------------------------------

def _read_lift_state(
    base_env,
    env_index: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Read cube, EE, robot-base, and goal-world positions from scene.

    Returns:
        cube_pos   : (3,) world pos of cube, or None
        ee_pos     : (3,) world pos of EE, or None
        robot_pos  : (3,) world pos of robot base, or None
        goal_world : (3,) goal world pos derived from command + robot base, or None
    """
    cube_pos   = None
    ee_pos     = None
    robot_pos  = None
    goal_world = None

    scene = getattr(base_env, "scene", None)
    if scene is None:
        return cube_pos, ee_pos, robot_pos, goal_world

    try:
        cube_pos = scene["object"].data.root_pos_w[env_index].detach().cpu().numpy()
    except Exception:
        pass

    try:
        ee_pos = scene["ee_frame"].data.target_pos_w[env_index, 0, :].detach().cpu().numpy()
    except Exception:
        pass

    try:
        robot_pos = scene["robot"].data.root_pos_w[env_index].detach().cpu().numpy()
    except Exception:
        pass

    try:
        cmd_mgr = getattr(base_env, "command_manager", None)
        if cmd_mgr is not None and robot_pos is not None:
            goal_robot = cmd_mgr.get_command("object_pose")[env_index, :3].detach().cpu().numpy()
            # Franka base has identity rotation → world goal = base + goal_in_base_frame
            goal_world = robot_pos + goal_robot
    except Exception:
        pass

    return cube_pos, ee_pos, robot_pos, goal_world


# ---------------------------------------------------------------------------
# Per-frame render
# ---------------------------------------------------------------------------

def _render_frame(
    cube_pos   : np.ndarray,
    ee_pos     : np.ndarray,
    robot_pos  : np.ndarray,
    goal_world : np.ndarray,
    reward     : float,
    done       : bool,
    lifted     : bool,
    frame_idx  : int,
    episode_idx: int,
    episode_step: int,
    cube_to_goal_dist: float,
    cube_to_ee_dist  : float,
    ee_trail_xz   : list[tuple[float, float]],
    cube_trail_xz : list[tuple[float, float]],
    ee_trail_xy   : list[tuple[float, float]],
    cube_trail_xy : list[tuple[float, float]],
    hist_dist     : list[float],
    hist_height   : list[float],
    width  : int,
    height : int,
) -> np.ndarray:
    image = Image.new("RGB", (width, height), _C_BG)
    draw  = ImageDraw.Draw(image)

    # ---- Telemetry box ----
    status      = "DONE" if done else ("LIFTED" if lifted else "REACH")
    status_col  = _C_TEXT_ERR if done else (_C_TEXT_OK if lifted else _C_TEXT)
    text_lines  = [
        f"status={status}   frame={frame_idx}   episode={episode_idx}   step={episode_step}",
        f"cube=({cube_pos[0]:+.3f}, {cube_pos[1]:+.3f}, z={cube_pos[2]:+.3f})   "
        f"height_above_table={cube_pos[2]:.3f}",
        f"ee  =({ee_pos[0]:+.3f}, {ee_pos[1]:+.3f}, z={ee_pos[2]:+.3f})   "
        f"cube↔ee={cube_to_ee_dist:.3f} m",
        f"goal=({goal_world[0]:+.3f}, {goal_world[1]:+.3f}, z={goal_world[2]:+.3f})   "
        f"cube↔goal={cube_to_goal_dist:.3f} m",
        f"reward={reward:+.4f}",
    ]

    tpad_x = 14
    tpad_y = 10
    line_h = 19
    tbox_top = 12
    tbox = (20, tbox_top, width - 20,
            tbox_top + tpad_y * 2 + line_h * len(text_lines) - 4)
    draw.rectangle(tbox, fill=_C_PANEL_BG, outline=(188, 194, 204), width=1)
    ty = tbox[1] + tpad_y
    for i, line in enumerate(text_lines):
        col = status_col if i == 0 else _C_TEXT
        draw.text((tbox[0] + tpad_x, ty), line, fill=col)
        ty += line_h

    # ---- Panel areas ----
    panel_top   = tbox[3] + 18
    half_w      = (width - 60) // 2
    chart_h     = 80
    panel_bot   = height - chart_h - 30
    left_panel  = (20,          panel_top, 20 + half_w,    panel_bot)
    right_panel = (40 + half_w, panel_top, 40 + 2 * half_w, panel_bot)

    for panel, label in [(left_panel, "Side view  (x – z)"),
                         (right_panel, "Top-down  (x – y)")]:
        draw.rectangle(panel, fill=_C_PANEL_BG, outline=_C_BORDER, width=2)
        draw.text((panel[0] + 6, panel_top - 17), label, fill=_C_TEXT)

    # ==================== LEFT PANEL: side view (x – z) ====================
    rb_x, rb_z = float(robot_pos[0]), 0.0          # robot base sits at z≈0
    cx_xz = float(robot_pos[0]) + 0.45             # centre x between base and typical goal
    cz_xz = 0.20                                   # centre z in view
    xz_bounds = _adaptive_bounds(
        [(rb_x, rb_z),
         (float(cube_pos[0]), float(cube_pos[2])),
         (float(ee_pos[0]), float(ee_pos[2])),
         (float(goal_world[0]), float(goal_world[2]))]
        + [(x, z) for x, z in cube_trail_xz[-30:]]
        + [(x, z) for x, z in ee_trail_xz[-30:]],
        min_x_span=_SIDE_X_SPAN,
        min_y_span=_SIDE_Z_SPAN,
    )

    # Table surface (z = 0)
    _draw_dashed_hline(draw, 0.0,  xz_bounds, left_panel, _C_TABLE, dash=14, gap=6)
    # Lift threshold (z = 0.04)
    _draw_dashed_hline(draw, 0.04, xz_bounds, left_panel, _C_LIFT_LINE, dash=6, gap=4)
    # Goal height
    _draw_dashed_hline(draw, float(goal_world[2]), xz_bounds, left_panel, (*_C_GOAL, ), dash=8, gap=5)

    # Arm stick + EE trail + cube trail
    _draw_arm_stick(draw, rb_x, rb_z, float(ee_pos[0]), float(ee_pos[2]), xz_bounds, left_panel)
    _draw_trail(draw, ee_trail_xz,   xz_bounds, left_panel, _C_EE_TRAIL,   width=2)
    _draw_trail(draw, cube_trail_xz, xz_bounds, left_panel, _C_CUBE_TRAIL, width=2)
    _draw_goal_marker(draw, float(goal_world[0]), float(goal_world[2]), xz_bounds, left_panel)
    _draw_cube(draw, float(cube_pos[0]), float(cube_pos[2]), xz_bounds, left_panel, lifted)
    _draw_ee(draw, float(ee_pos[0]), float(ee_pos[2]), xz_bounds, left_panel)

    # Axis labels
    for z_label, z_val in [("+0.5 m", 0.5), ("+0.25 m", 0.25), ("table", 0.0)]:
        px, py = _project(xz_bounds[0] + 0.01, z_val, xz_bounds, left_panel)
        if left_panel[1] < py < left_panel[3]:
            draw.text((left_panel[0] + 4, py - 8), z_label, fill=(120, 120, 120))

    # ==================== RIGHT PANEL: top-down (x – y) ====================
    xy_bounds = _adaptive_bounds(
        [(float(cube_pos[0]), float(cube_pos[1])),
         (float(ee_pos[0]), float(ee_pos[1])),
         (float(goal_world[0]), float(goal_world[1]))]
        + [(x, y) for x, y in cube_trail_xy[-30:]]
        + [(x, y) for x, y in ee_trail_xy[-30:]],
        min_x_span=_TOP_X_SPAN,
        min_y_span=_TOP_Y_SPAN,
    )

    # Robot base dot
    rbx, rby = _project(float(robot_pos[0]), float(robot_pos[1]), xy_bounds, right_panel)
    draw.ellipse([rbx - 6, rby - 6, rbx + 6, rby + 6], fill=_C_ROBOT_BASE)

    _draw_trail(draw, ee_trail_xy,   xy_bounds, right_panel, _C_EE_TRAIL,   width=2)
    _draw_trail(draw, cube_trail_xy, xy_bounds, right_panel, _C_CUBE_TRAIL, width=2)
    _draw_goal_marker(draw, float(goal_world[0]), float(goal_world[1]), xy_bounds, right_panel)
    _draw_cube(draw, float(cube_pos[0]), float(cube_pos[1]), xy_bounds, right_panel, lifted)
    _draw_ee(draw, float(ee_pos[0]), float(ee_pos[1]), xy_bounds, right_panel)

    # Axis labels inside right panel
    draw.text((right_panel[0] + 4, right_panel[3] - 16), "← y →", fill=(140, 140, 140))
    draw.text((right_panel[2] - 28, right_panel[3] - 16), "x →", fill=(140, 140, 140))

    # ==================== BOTTOM: mini chart strips ====================
    chart_top  = panel_bot + 12
    chart_bot  = height - 12
    mid        = width // 2
    dist_panel = (20, chart_top, mid - 10, chart_bot)
    hgt_panel  = (mid + 10, chart_top, width - 20, chart_bot)

    for panel, label in [(dist_panel, "cube↔goal dist (m)"),
                         (hgt_panel,  "cube height z (m)")]:
        draw.rectangle(panel, fill=_C_PANEL_BG, outline=_C_BORDER, width=1)
        draw.text((panel[0] + 4, panel[0] + 2 - panel[0] + chart_top - 2), label, fill=_C_TEXT)

    def _draw_chart(data: list[float], panel: tuple, color: tuple, y_min: float, y_max: float):
        if len(data) < 2:
            return
        n = len(data)
        pw = panel[2] - panel[0]
        ph = panel[3] - panel[1]
        span = max(1e-6, y_max - y_min)
        pts = []
        for i, v in enumerate(data):
            px = panel[0] + int(i / (n - 1) * pw)
            py = panel[3] - int((v - y_min) / span * ph)
            py = max(panel[1], min(panel[3], py))
            pts.append((px, py))
        draw.line(pts, fill=color, width=2)
        # Most-recent value annotated
        last_x, last_y = pts[-1]
        draw.text((last_x - 30, last_y - 12), f"{data[-1]:.3f}", fill=color)

    _draw_chart(hist_dist,   dist_panel, _C_GOAL,  0.0, max(0.1, max(hist_dist) if hist_dist else 0.1))
    _draw_chart(hist_height, hgt_panel,  _C_CUBE,  0.0, max(0.05, max(hist_height) if hist_height else 0.05))

    # ---- Legend ----
    lx = width - 180
    ly = panel_top + 8
    for marker_color, lbl in [(_C_EE, "End-effector"),
                               (_C_CUBE, "Cube"),
                               (_C_GOAL, "Goal")]:
        draw.ellipse([lx, ly, lx + 10, ly + 10], fill=marker_color)
        draw.text((lx + 14, ly - 1), lbl, fill=_C_TEXT)
        ly += 18
    draw.text((lx, ly), "-- table z=0", fill=_C_TABLE)
    ly += 18
    draw.text((lx, ly), "-- lift threshold", fill=_C_LIFT_LINE)

    return np.asarray(image, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args_cli: argparse.Namespace) -> None:
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
    from isaaclab.envs import ManagerBasedRLEnvCfg
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(args_cli.task)
    env_cfg.sim.device = device
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1
    # Disable observation noise during recording
    env_cfg.observations.policy.enable_corruption = False

    env = gym.make(args_cli.task, cfg=env_cfg)

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.output_file)), exist_ok=True)
    writer = imageio.get_writer(args_cli.output_file, fps=args_cli.fps)

    env_index  = max(0, args_cli.env_index)
    frame_idx  = 0
    episode_idx   = 0
    episode_step  = 0

    # Trails and history buffers
    ee_trail_xz  : list[tuple[float, float]] = []
    cube_trail_xz: list[tuple[float, float]] = []
    ee_trail_xy  : list[tuple[float, float]] = []
    cube_trail_xy: list[tuple[float, float]] = []
    hist_dist    : list[float] = []
    hist_height  : list[float] = []

    # Fallback pose when scene attributes are not available
    _fallback = {
        "cube_pos":  np.array([0.5, 0.0, 0.055]),
        "ee_pos":    np.array([0.5, 0.0, 0.4]),
        "robot_pos": np.zeros(3),
        "goal":      np.array([0.5, 0.0, 0.35]),
    }

    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Checkpoint: {args_cli.checkpoint}")
    print(f"[INFO] Num envs: {env_cfg.scene.num_envs}")
    print(f"[INFO] Output: {args_cli.output_file}")

    try:
        if args_cli.rl_library == "rsl_rl":
            from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
            from rsl_rl.runners import OnPolicyRunner

            agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(
                args_cli.task, "rsl_rl_cfg_entry_point"
            )
            agent_cfg.device = device

            env_wrapped = RslRlVecEnvWrapper(env)
            runner = OnPolicyRunner(
                env_wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
            )
            runner.load(args_cli.checkpoint)
            policy = runner.get_inference_policy(device=env_wrapped.unwrapped.device)
            obs    = env_wrapped.get_observations()

            while simulation_app.is_running():
                with torch.inference_mode():
                    actions = policy(obs)
                    obs, rewards, dones, _ = env_wrapped.step(actions)

                # --- Unwrap obs ---
                policy_obs = _extract_policy_obs(obs)
                if policy_obs.ndim == 1:
                    policy_obs = policy_obs.unsqueeze(0)
                active = min(env_index, policy_obs.shape[0] - 1)
                reward = float(rewards[active].item())
                done   = bool(dones[active].item())

                # --- Read scene state ---
                base_env = env_wrapped.unwrapped
                cube_pos, ee_pos, robot_pos, goal_world = _read_lift_state(base_env, active)
                cube_pos   = cube_pos   if cube_pos   is not None else _fallback["cube_pos"]
                ee_pos     = ee_pos     if ee_pos     is not None else _fallback["ee_pos"]
                robot_pos  = robot_pos  if robot_pos  is not None else _fallback["robot_pos"]
                goal_world = goal_world if goal_world is not None else _fallback["goal"]

                lifted = bool(cube_pos[2] > 0.04)
                cube_to_goal = float(np.linalg.norm(goal_world - cube_pos))
                cube_to_ee   = float(np.linalg.norm(ee_pos - cube_pos))

                # --- Update trails ---
                ee_trail_xz.append(  (float(ee_pos[0]),   float(ee_pos[2])))
                cube_trail_xz.append((float(cube_pos[0]), float(cube_pos[2])))
                ee_trail_xy.append(  (float(ee_pos[0]),   float(ee_pos[1])))
                cube_trail_xy.append((float(cube_pos[0]), float(cube_pos[1])))
                hist_dist.append(cube_to_goal)
                hist_height.append(float(cube_pos[2]))

                tl = args_cli.trail_length
                ch = args_cli.chart_history
                if len(ee_trail_xz) > tl:
                    ee_trail_xz   = ee_trail_xz[-tl:]
                    cube_trail_xz = cube_trail_xz[-tl:]
                    ee_trail_xy   = ee_trail_xy[-tl:]
                    cube_trail_xy = cube_trail_xy[-tl:]
                if len(hist_dist) > ch:
                    hist_dist   = hist_dist[-ch:]
                    hist_height = hist_height[-ch:]

                # --- Render ---
                frame = _render_frame(
                    cube_pos=cube_pos,
                    ee_pos=ee_pos,
                    robot_pos=robot_pos,
                    goal_world=goal_world,
                    reward=reward,
                    done=done,
                    lifted=lifted,
                    frame_idx=frame_idx,
                    episode_idx=episode_idx,
                    episode_step=episode_step,
                    cube_to_goal_dist=cube_to_goal,
                    cube_to_ee_dist=cube_to_ee,
                    ee_trail_xz=ee_trail_xz,
                    cube_trail_xz=cube_trail_xz,
                    ee_trail_xy=ee_trail_xy,
                    cube_trail_xy=cube_trail_xy,
                    hist_dist=hist_dist,
                    hist_height=hist_height,
                    width=args_cli.frame_width,
                    height=args_cli.frame_height,
                )
                writer.append_data(frame)

                frame_idx    += 1
                episode_step += 1
                if done:
                    episode_idx  += 1
                    episode_step  = 0
                    ee_trail_xz.clear()
                    cube_trail_xz.clear()
                    ee_trail_xy.clear()
                    cube_trail_xy.clear()

                if frame_idx % 50 == 0:
                    print(
                        f"[INFO] frame={frame_idx}  episode={episode_idx}"
                        f"  cube_z={cube_pos[2]:.3f}  dist_to_goal={cube_to_goal:.3f}"
                    )

                if args_cli.max_frames > 0 and frame_idx >= args_cli.max_frames:
                    break
                if args_cli.num_episodes > 0 and episode_idx >= args_cli.num_episodes:
                    break

        elif args_cli.rl_library == "rl_games":
            import math as _math
            from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
            from rl_games.common import env_configurations, vecenv
            from rl_games.common.algo_observer import IsaacAlgoObserver
            from rl_games.torch_runner import Runner

            agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")
            agent_cfg["params"]["load_checkpoint"] = True
            agent_cfg["params"]["load_path"] = args_cli.checkpoint
            agent_cfg["params"]["config"]["device"] = device
            agent_cfg["params"]["config"]["device_name"] = device
            clip_obs = agent_cfg["params"]["env"].get("clip_observations", _math.inf)
            clip_acts = agent_cfg["params"]["env"].get("clip_actions", _math.inf)

            env_wrapped = RlGamesVecEnvWrapper(env, device, clip_obs, clip_acts)
            vecenv.register("IsaacRlgWrapper",
                            lambda cfg, n, **kw: RlGamesGpuEnv(cfg, n, **kw))
            env_configurations.register(
                "rlgpu", {"vecenv_type": "IsaacRlgWrapper",
                          "env_creator": lambda **kw: env_wrapped})
            agent_cfg["params"]["config"]["num_actors"] = env_wrapped.unwrapped.num_envs

            runner = Runner(IsaacAlgoObserver())
            runner.load(agent_cfg)

            from rl_games.common.player import BasePlayer
            agent: BasePlayer = runner.create_player()
            agent.restore(args_cli.checkpoint)
            agent.reset()

            obs = env_wrapped.reset()
            if isinstance(obs, dict) and "obs" in obs:
                obs = obs["obs"]
            _ = agent.get_batch_size(obs, 1)
            if agent.is_rnn:
                agent.init_rnn()

            while simulation_app.is_running():
                with torch.inference_mode():
                    actor_obs = agent.obs_to_torch(obs)
                    actions   = agent.get_action(actor_obs, is_deterministic=True)
                    next_obs, rewards, dones, _ = env_wrapped.step(actions)

                policy_obs = _extract_policy_obs(next_obs)
                if policy_obs.ndim == 1:
                    policy_obs = policy_obs.unsqueeze(0)
                active = min(env_index, policy_obs.shape[0] - 1)
                reward = float(rewards[active].item())
                done   = bool(dones[active].item())

                base_env = env_wrapped.unwrapped
                cube_pos, ee_pos, robot_pos, goal_world = _read_lift_state(base_env, active)
                cube_pos   = cube_pos   if cube_pos   is not None else _fallback["cube_pos"]
                ee_pos     = ee_pos     if ee_pos     is not None else _fallback["ee_pos"]
                robot_pos  = robot_pos  if robot_pos  is not None else _fallback["robot_pos"]
                goal_world = goal_world if goal_world is not None else _fallback["goal"]

                lifted = bool(cube_pos[2] > 0.04)
                cube_to_goal = float(np.linalg.norm(goal_world - cube_pos))
                cube_to_ee   = float(np.linalg.norm(ee_pos - cube_pos))

                ee_trail_xz.append(  (float(ee_pos[0]),   float(ee_pos[2])))
                cube_trail_xz.append((float(cube_pos[0]), float(cube_pos[2])))
                ee_trail_xy.append(  (float(ee_pos[0]),   float(ee_pos[1])))
                cube_trail_xy.append((float(cube_pos[0]), float(cube_pos[1])))
                hist_dist.append(cube_to_goal)
                hist_height.append(float(cube_pos[2]))

                tl = args_cli.trail_length
                ch = args_cli.chart_history
                if len(ee_trail_xz) > tl:
                    ee_trail_xz = ee_trail_xz[-tl:]; cube_trail_xz = cube_trail_xz[-tl:]
                    ee_trail_xy = ee_trail_xy[-tl:]; cube_trail_xy = cube_trail_xy[-tl:]
                if len(hist_dist) > ch:
                    hist_dist = hist_dist[-ch:]; hist_height = hist_height[-ch:]

                frame = _render_frame(
                    cube_pos=cube_pos, ee_pos=ee_pos, robot_pos=robot_pos,
                    goal_world=goal_world, reward=reward, done=done, lifted=lifted,
                    frame_idx=frame_idx, episode_idx=episode_idx,
                    episode_step=episode_step,
                    cube_to_goal_dist=cube_to_goal, cube_to_ee_dist=cube_to_ee,
                    ee_trail_xz=ee_trail_xz, cube_trail_xz=cube_trail_xz,
                    ee_trail_xy=ee_trail_xy, cube_trail_xy=cube_trail_xy,
                    hist_dist=hist_dist, hist_height=hist_height,
                    width=args_cli.frame_width, height=args_cli.frame_height,
                )
                writer.append_data(frame)

                frame_idx += 1; episode_step += 1
                if done:
                    episode_idx += 1; episode_step = 0
                    ee_trail_xz.clear(); cube_trail_xz.clear()
                    ee_trail_xy.clear(); cube_trail_xy.clear()

                if frame_idx % 50 == 0:
                    print(f"[INFO] frame={frame_idx}  episode={episode_idx}"
                          f"  cube_z={cube_pos[2]:.3f}  dist_to_goal={cube_to_goal:.3f}")

                if agent.is_rnn and agent.states is not None:
                    for st in agent.states:
                        st[:, dones, :] = 0.0

                obs = next_obs["obs"] if isinstance(next_obs, dict) and "obs" in next_obs else next_obs

                if args_cli.max_frames > 0 and frame_idx >= args_cli.max_frames:
                    break
                if args_cli.num_episodes > 0 and episode_idx >= args_cli.num_episodes:
                    break

        else:
            raise ValueError(f"Unsupported --rl_library: {args_cli.rl_library!r}")

    finally:
        writer.close()
        env.close()
        simulation_app.close()
        print(f"[INFO] Recording complete: {frame_idx} frames, {episode_idx} episodes.")
        print(f"[INFO] Saved → {args_cli.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Record Isaac-Lift-Cube-Franka-v0 rollouts as a 2D fallback MP4 "
            "(no Isaac renderer required)."
        )
    )
    parser.add_argument(
        "--task", type=str, default="Isaac-Lift-Cube-Franka-v0",
        help="Gym task id.",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the RSL-RL / rl-games checkpoint file.",
    )
    parser.add_argument(
        "--rl_library", type=str, default="rsl_rl", choices=["rsl_rl", "rl_games"],
        help="RL library used to train the checkpoint.",
    )
    parser.add_argument(
        "--num_envs", type=int, default=1,
        help="Number of parallel environments (only env-0 is visualised).",
    )
    parser.add_argument(
        "--env_index", type=int, default=0,
        help="Which environment index to visualise.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Compute device ('cuda' auto-selects the freest GPU).",
    )
    parser.add_argument(
        "--headless", action="store_true", default=True,
        help="Run without a display (default: True for cluster use).",
    )
    parser.add_argument(
        "--output_file", type=str,
        default="./recordings/lift_franka_fallback.mp4",
        help="Output MP4 file path.",
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Output video frame rate.",
    )
    parser.add_argument(
        "--max_frames", type=int, default=1500,
        help="Max frames to record (<=0 = unlimited).",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=3,
        help="Stop after this many complete episodes (<=0 = unlimited).",
    )
    parser.add_argument(
        "--trail_length", type=int, default=150,
        help="Number of past positions to draw as trajectory trail.",
    )
    parser.add_argument(
        "--chart_history", type=int, default=300,
        help="Number of steps shown in the bottom metric charts.",
    )
    parser.add_argument(
        "--frame_width",  type=int, default=1280,
        help="Output video width in pixels.",
    )
    parser.add_argument(
        "--frame_height", type=int, default=720,
        help="Output video height in pixels.",
    )
    args_cli = parser.parse_args()
    main(args_cli)
