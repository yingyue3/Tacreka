# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

"""Record Humanoid policy rollouts without Isaac camera rendering.

This script runs policy inference headlessly and generates a lightweight 2D MP4
visualization from simulation states and observations. It is intended for
clusters where Isaac renderer-based recording is unavailable.

Humanoid observation layout (75 dims, from LocomotionEnv._get_observations):
  obs[ 0]      : torso_z (height above world origin)
  obs[ 1: 4]   : vel_loc (local linear velocity x, y, z)
  obs[ 4: 7]   : angvel_loc * angular_velocity_scale
  obs[ 7]      : yaw  (normalized)
  obs[ 8]      : roll (normalized)
  obs[ 9]      : angle_to_target (signed)
  obs[10]      : up_proj   (1 = fully upright)
  obs[11]      : heading_proj (1 = perfectly aligned with target)
  obs[12:33]   : dof_pos_scaled (21 joints)
  obs[33:54]   : dof_vel * dof_vel_scale (21 joints)
  obs[54:75]   : actions (21)
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
# Body-segment lengths (metres, MuJoCo humanoid approximations)
# ---------------------------------------------------------------------------
_SEG_TORSO     = 0.33   # hip → chest
_SEG_HEAD_R    = 0.085  # head radius
_SEG_UPPER_ARM = 0.27
_SEG_LOWER_ARM = 0.22
_SEG_THIGH     = 0.34
_SEG_SHIN      = 0.30
_SEG_FOOT      = 0.10

# DOF indices (0-based within the 21-DOF humanoid array).
# Order follows joint_gears in HumanoidEnvCfg:
#   0-1  lower_waist (z-twist, y-pitch)
#   2-3  right_upper_arm,  4-5 left_upper_arm
#   6    pelvis,  7 right_lower_arm,  8 left_lower_arm
#   9-11 right_thigh (x-abduct, y-flex, z-twist)
#  12-14 left_thigh  (x-abduct, y-flex, z-twist)
#  15   right_knee,  16 left_knee
#  17-18 right_foot (x-pitch, y-eversion)
#  19-20 left_foot  (x-pitch, y-eversion)
_D_WAIST_Y  = 1
_D_R_SHLDR  = 3
_D_L_SHLDR  = 5
_D_R_ELBOW  = 7
_D_L_ELBOW  = 8
_D_R_HIP_X  = 9   # abduction
_D_R_HIP_Y  = 10  # sagittal flex
_D_L_HIP_X  = 12  # abduction
_D_L_HIP_Y  = 13  # sagittal flex
_D_R_KNEE   = 15
_D_L_KNEE   = 16
_D_R_ANKLE  = 17  # pitch
_D_L_ANKLE  = 19  # pitch


# ---------------------------------------------------------------------------
# Shared helpers (identical to quadcopter fallback)
# ---------------------------------------------------------------------------

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


def _project_to_panel(
    x: float,
    y: float,
    bounds: tuple[float, float, float, float],
    panel: tuple[int, int, int, int],
) -> tuple[int, int]:
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
    cx = 0.5 * (min(xs) + max(xs))
    cy = 0.5 * (min(ys) + max(ys))
    span_x = max(min_span, max(xs) - min(xs))
    span_y = max(min_span, max(ys) - min(ys))
    x_half = 0.5 * span_x * (1.0 + margin_ratio)
    y_half = 0.5 * span_y * (1.0 + margin_ratio)
    return cx - x_half, cx + x_half, cy - y_half, cy + y_half


def _draw_trail(
    draw: ImageDraw.ImageDraw,
    trail: list[tuple[float, float]],
    bounds: tuple[float, float, float, float],
    panel: tuple[int, int, int, int],
    color: tuple[int, int, int],
    width: int = 2,
):
    if len(trail) < 2:
        return
    points = [_project_to_panel(x, y, bounds, panel) for (x, y) in trail]
    draw.line(points, fill=color, width=width)


# ---------------------------------------------------------------------------
# Humanoid forward kinematics (sagittal-plane, for side view)
# ---------------------------------------------------------------------------

def _stick_figure_xz(pos_w: np.ndarray, dof_scaled: np.ndarray) -> dict:
    """Compute approximate 2D joint positions in world (x, z) for a sagittal stick figure.

    Uses a simplified planar FK.  dof_scaled values are obs[12:33], scaled to [-1, 1].
    All returned values are np.ndarray of shape (2,) = [world_x, world_z].
    """
    tx, tz = float(pos_w[0]), float(pos_w[2])
    d = dof_scaled  # 21-element array

    # ---- angles (radians, approximate, for visualisation only) ----
    # waist sagittal lean: typical range ±0.785 rad
    waist_pitch  = float(d[_D_WAIST_Y])  * 0.60
    # hip flex/extend: typical range ±1.0 rad
    r_hip        = float(d[_D_R_HIP_Y])  * 0.90
    l_hip        = float(d[_D_L_HIP_Y])  * 0.90
    # knee: actual range [-1.57, 0.1]; midpoint at ~-0.73 rad
    r_knee       = float(d[_D_R_KNEE])   * 0.74 - 0.73  # always flexed
    l_knee       = float(d[_D_L_KNEE])   * 0.74 - 0.73
    # shoulder flex: ±0.785 rad
    r_shldr      = float(d[_D_R_SHLDR])  * 0.70
    l_shldr      = float(d[_D_L_SHLDR])  * 0.70
    # elbow flex: ±0.785 rad
    r_elbow      = float(d[_D_R_ELBOW])  * 0.70
    l_elbow      = float(d[_D_L_ELBOW])  * 0.70

    # unit direction from angle measured from -z axis (0 = hanging straight down)
    def fwd(angle: float) -> np.ndarray:
        return np.array([math.sin(angle), -math.cos(angle)], dtype=float)

    spine_dir = np.array([math.sin(waist_pitch), math.cos(waist_pitch)], dtype=float)
    root  = np.array([tx, tz], dtype=float)
    hip   = root - spine_dir * (_SEG_TORSO * 0.45)
    chest = root + spine_dir * (_SEG_TORSO * 0.55)
    neck  = chest + spine_dir * 0.08
    head  = neck  + spine_dir * (_SEG_HEAD_R + 0.02)

    # --- Legs ---
    # absolute thigh angle from vertical = hip flex + waist lean
    r_thigh_abs = r_hip + waist_pitch
    r_knee_pt   = hip   + _SEG_THIGH * fwd(r_thigh_abs)
    r_shin_abs  = r_thigh_abs + r_knee          # knee_flex is negative
    r_ankle_pt  = r_knee_pt   + _SEG_SHIN  * fwd(r_shin_abs)
    r_toe_pt    = r_ankle_pt  + np.array([_SEG_FOOT, 0.0])

    l_thigh_abs = l_hip + waist_pitch
    l_knee_pt   = hip   + _SEG_THIGH * fwd(l_thigh_abs)
    l_shin_abs  = l_thigh_abs + l_knee
    l_ankle_pt  = l_knee_pt   + _SEG_SHIN  * fwd(l_shin_abs)
    l_toe_pt    = l_ankle_pt  + np.array([_SEG_FOOT, 0.0])

    # --- Arms ---
    # shoulders are slightly offset from chest in the x-direction for separation
    r_shldr_pt  = chest + np.array([ 0.08, 0.0])
    l_shldr_pt  = chest + np.array([-0.08, 0.0])
    r_arm_abs   = r_shldr + waist_pitch
    l_arm_abs   = l_shldr + waist_pitch
    r_elbow_pt  = r_shldr_pt + _SEG_UPPER_ARM * fwd(r_arm_abs)
    l_elbow_pt  = l_shldr_pt + _SEG_UPPER_ARM * fwd(l_arm_abs)
    r_hand_pt   = r_elbow_pt  + _SEG_LOWER_ARM * fwd(r_arm_abs + r_elbow)
    l_hand_pt   = l_elbow_pt  + _SEG_LOWER_ARM * fwd(l_arm_abs + l_elbow)

    return {
        "hip": hip, "root": root, "chest": chest, "neck": neck, "head": head,
        "r_knee": r_knee_pt, "r_ankle": r_ankle_pt, "r_toe": r_toe_pt,
        "l_knee": l_knee_pt, "l_ankle": l_ankle_pt, "l_toe": l_toe_pt,
        "r_shoulder": r_shldr_pt, "r_elbow": r_elbow_pt, "r_hand": r_hand_pt,
        "l_shoulder": l_shldr_pt, "l_elbow": l_elbow_pt, "l_hand": l_hand_pt,
    }


# ---------------------------------------------------------------------------
# Humanoid-specific drawing helpers
# ---------------------------------------------------------------------------

def _draw_ground_line(
    draw: ImageDraw.ImageDraw,
    termination_height: float,
    bounds: tuple[float, float, float, float],
    panel: tuple[int, int, int, int],
):
    """Draw a dashed ground / termination-height reference line in the side panel."""
    x_min, x_max, _, _ = bounds
    px0, _, px1, _ = panel
    gx0, gy = _project_to_panel(x_min, termination_height, bounds, panel)
    gx1, _ = _project_to_panel(x_max, termination_height, bounds, panel)
    # dashed line
    dash_len = 12
    gap_len = 6
    x = gx0
    while x < gx1:
        draw.line([(x, gy), (min(x + dash_len, gx1), gy)], fill=(200, 80, 80), width=1)
        x += dash_len + gap_len


def _draw_humanoid_side(
    draw: ImageDraw.ImageDraw,
    pos_w: np.ndarray,
    dof_scaled: np.ndarray,
    bounds: tuple[float, float, float, float],
    panel: tuple[int, int, int, int],
    fallen: bool,
):
    """Draw a full 2D stick figure in the side (x-z) panel using sagittal-plane FK."""
    joints = _stick_figure_xz(pos_w, dof_scaled)

    # Project a world-(x,z) point to panel pixel.
    def pp(pt: np.ndarray) -> tuple[int, int]:
        return _project_to_panel(float(pt[0]), float(pt[1]), bounds, panel)

    # Colour scheme: blue tones when running, red when fallen
    spine_col  = (200,  60,  60) if fallen else ( 38, 132, 255)
    r_col      = (220, 100,  40) if fallen else ( 30, 180,  80)   # right limb
    l_col      = (180,  60, 120) if fallen else ( 30, 120, 220)   # left limb
    head_fill  = (220,  80,  80) if fallen else ( 90, 190, 255)
    joint_fill = (200, 200, 200)
    lw = 3

    # ---- Left limbs (drawn first so right limbs appear on top) ----
    # Left leg
    draw.line([pp(joints["hip"]),      pp(joints["l_knee"])],  fill=l_col, width=lw)
    draw.line([pp(joints["l_knee"]),   pp(joints["l_ankle"])], fill=l_col, width=lw)
    draw.line([pp(joints["l_ankle"]),  pp(joints["l_toe"])],   fill=l_col, width=lw)
    # Left arm
    draw.line([pp(joints["l_shoulder"]), pp(joints["l_elbow"])], fill=l_col, width=lw)
    draw.line([pp(joints["l_elbow"]),    pp(joints["l_hand"])],  fill=l_col, width=lw)

    # ---- Spine (hip → chest → neck) ----
    draw.line([pp(joints["hip"]),   pp(joints["chest"])], fill=spine_col, width=lw + 1)
    draw.line([pp(joints["chest"]), pp(joints["neck"])],  fill=spine_col, width=lw)

    # ---- Right limbs ----
    # Right leg
    draw.line([pp(joints["hip"]),     pp(joints["r_knee"])],  fill=r_col, width=lw)
    draw.line([pp(joints["r_knee"]),  pp(joints["r_ankle"])], fill=r_col, width=lw)
    draw.line([pp(joints["r_ankle"]), pp(joints["r_toe"])],   fill=r_col, width=lw)
    # Right arm
    draw.line([pp(joints["r_shoulder"]), pp(joints["r_elbow"])], fill=r_col, width=lw)
    draw.line([pp(joints["r_elbow"]),    pp(joints["r_hand"])],  fill=r_col, width=lw)

    # ---- Head ----
    hx, hz = pp(joints["head"])
    # Map head radius from world metres to panel pixels
    span_world = max(1e-6, bounds[1] - bounds[0])
    span_px    = max(1, panel[2] - panel[0])
    r_px = int(_SEG_HEAD_R / span_world * span_px)
    r_px = max(5, min(r_px, 22))
    draw.ellipse([hx - r_px, hz - r_px, hx + r_px, hz + r_px],
                 fill=head_fill, outline=spine_col, width=2)

    # ---- Joint dots ----
    jr = 3
    for key in ("r_knee", "l_knee", "r_ankle", "l_ankle", "r_elbow", "l_elbow"):
        jx, jz = pp(joints[key])
        draw.ellipse([jx - jr, jz - jr, jx + jr, jz + jr], fill=joint_fill)


def _draw_humanoid_topdown(
    draw: ImageDraw.ImageDraw,
    pos_w: np.ndarray,
    dof_scaled: np.ndarray,
    yaw: float,
    bounds: tuple[float, float, float, float],
    panel: tuple[int, int, int, int],
    fallen: bool,
):
    """Draw a top-down (x-y) humanoid silhouette: torso pill, head, arms, foot circles."""
    x, y = float(pos_w[0]), float(pos_w[1])

    # Scale metres → pixels in panel
    span_wx = max(1e-6, bounds[1] - bounds[0])
    span_wy = max(1e-6, bounds[3] - bounds[2])
    span_px = max(1, panel[2] - panel[0])
    span_py = max(1, panel[3] - panel[1])
    m2px_x = span_px / span_wx
    m2px_y = span_py / span_wy

    def w2p(wx: float, wy: float) -> tuple[int, int]:
        return _project_to_panel(wx, wy, bounds, panel)

    def offset_px(dx_world: float, dy_world: float) -> tuple[int, int]:
        """Rotate a body-frame offset (forward=+x, left=+y) into world frame then to pixels."""
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        wx = x + cos_y * dx_world - sin_y * dy_world
        wy = y + sin_y * dx_world + cos_y * dy_world
        return w2p(wx, wy)

    body_col   = (200,  60,  60) if fallen else ( 38, 132, 255)
    outline_col= ( 80,  20,  20) if fallen else ( 10,  60, 120)
    r_col      = (220, 100,  40) if fallen else ( 30, 180,  80)
    l_col      = (180,  60, 120) if fallen else ( 30, 120, 220)
    head_col   = (220,  80,  80) if fallen else ( 90, 190, 255)

    # ---- Torso: elongated ellipse aligned with yaw ----
    torso_half_len = _SEG_TORSO * 0.55 * m2px_x
    torso_half_wid = 0.10      * m2px_x   # approx shoulder width / 2

    cx, cy = w2p(x, y)
    # Approximate with a rotated rectangle (PIL has no native rotated ellipse easily)
    n  = 24
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    angles = [2 * math.pi * i / n for i in range(n)]
    torso_poly = []
    for a in angles:
        lx = torso_half_len * math.cos(a)
        ly = torso_half_wid * math.sin(a)
        rx = lx * cos_y - ly * sin_y
        ry = lx * sin_y + ly * cos_y
        px = cx + rx
        py = cy - ry
        torso_poly.append((int(px), int(py)))
    draw.polygon(torso_poly, fill=body_col, outline=outline_col)

    # ---- Head ----
    head_r_px = int(_SEG_HEAD_R * m2px_x)
    head_r_px = max(4, min(head_r_px, 18))
    hx, hy = offset_px(_SEG_TORSO * 0.55 + _SEG_HEAD_R, 0.0)
    draw.ellipse([hx - head_r_px, hy - head_r_px, hx + head_r_px, hy + head_r_px],
                 fill=head_col, outline=outline_col, width=2)

    # ---- Foot circles (based on hip abduction and a half-thigh offset back) ----
    r_abd = float(dof_scaled[_D_R_HIP_X]) * 0.52    # right abducts in -y body frame
    l_abd = float(dof_scaled[_D_L_HIP_X]) * 0.52    # left  abducts in +y body frame
    foot_r_px = max(3, int(0.06 * m2px_x))
    for dy_body, col, abd in [(-0.10 - r_abd * 0.15, r_col, r_abd),
                               ( 0.10 + l_abd * 0.15, l_col, l_abd)]:
        fx, fy = offset_px(-_SEG_TORSO * 0.40, dy_body)
        draw.ellipse([fx - foot_r_px, fy - foot_r_px, fx + foot_r_px, fy + foot_r_px],
                     fill=col, outline=outline_col, width=1)

    # ---- Arms (short lines from shoulder to elbow) ----
    r_shldr_flex = float(dof_scaled[_D_R_SHLDR]) * 0.60
    l_shldr_flex = float(dof_scaled[_D_L_SHLDR]) * 0.60
    # In top-down, arm swings project mostly forward/backward; abduction is the lateral spread.
    # Visualise as short stubs to left/right of upper chest.
    arm_stub = _SEG_UPPER_ARM * 0.6
    for dy_body, flex, col in [(-0.15, r_shldr_flex, r_col),
                                ( 0.15, l_shldr_flex, l_col)]:
        sx, sy = offset_px(_SEG_TORSO * 0.30, dy_body)
        ex, ey = offset_px(_SEG_TORSO * 0.30 + arm_stub * math.sin(flex),
                           dy_body - arm_stub * 0.4 * math.cos(flex))
        draw.line([(sx, sy), (ex, ey)], fill=col, width=3)


def _draw_direction_arrow(
    draw: ImageDraw.ImageDraw,
    panel: tuple[int, int, int, int],
):
    """Draw a small +x direction indicator in the top-right corner of a panel."""
    px0, py0, px1, py1 = panel
    ax = px1 - 30
    ay = py0 + 20
    draw.line([(ax - 20, ay), (ax, ay)], fill=(80, 160, 80), width=2)
    draw.polygon([(ax, ay - 4), (ax, ay + 4), (ax + 8, ay)], fill=(80, 160, 80))
    draw.text((ax - 34, ay - 8), "+x", fill=(80, 160, 80))


# ---------------------------------------------------------------------------
# Velocity bar chart helper (right panel extra info)
# ---------------------------------------------------------------------------

def _draw_velocity_bars(
    draw: ImageDraw.ImageDraw,
    fwd_vel: float,
    lat_vel: float,
    up_proj: float,
    heading_proj: float,
    panel: tuple[int, int, int, int],
):
    """Draw four horizontal bars inside the right panel."""
    px0, py0, px1, py1 = panel
    bar_h = 14
    bar_max_w = (px1 - px0) - 60
    pad_x = px0 + 8
    entries = [
        ("fwd_vel", fwd_vel, 8.0, (60, 160, 60), (200, 60, 60)),
        ("lat_vel", lat_vel, 4.0, (60, 120, 220), (220, 120, 60)),
        ("up_proj", up_proj, 1.0, (80, 200, 80), (200, 80, 80)),
        ("hdg_proj", heading_proj, 1.0, (100, 160, 240), (240, 100, 100)),
    ]
    y = py0 + 12
    for label, val, max_val, pos_color, neg_color in entries:
        frac = max(-1.0, min(1.0, val / max(1e-6, max_val)))
        bar_color = pos_color if frac >= 0 else neg_color
        bar_w = int(abs(frac) * bar_max_w)
        bar_x0 = pad_x if frac >= 0 else pad_x - bar_w
        draw.rectangle([bar_x0, y, bar_x0 + bar_w, y + bar_h], fill=bar_color)
        draw.text((pad_x, y), f"{label}: {val:+.2f}", fill=(20, 20, 20))
        y += bar_h + 6


# ---------------------------------------------------------------------------
# Full frame renderer
# ---------------------------------------------------------------------------

def _render_humanoid_frame(
    pos_w: np.ndarray,
    lin_vel_w: np.ndarray,
    ang_vel_w: np.ndarray,
    heading_proj: float,
    up_proj: float,
    angle_to_target: float,
    torso_z_obs: float,
    dof_scaled: np.ndarray,
    reward: float,
    done: bool,
    fallen: bool,
    frame_idx: int,
    episode_idx: int,
    episode_step: int,
    yaw_est: float,
    termination_height: float,
    trail_xz: list[tuple[float, float]],
    trail_xy: list[tuple[float, float]],
    episode_max_x: float,
    width: int,
    height: int,
) -> np.ndarray:
    image = Image.new("RGB", (width, height), (246, 247, 250))
    draw = ImageDraw.Draw(image)

    fwd_vel = float(lin_vel_w[0])
    lat_vel = float(lin_vel_w[1])

    # status
    if fallen:
        status = "FALLEN"
        status_color = (200, 30, 30)
    elif done:
        status = "TIMEOUT"
        status_color = (150, 100, 0)
    else:
        status = "RUN"
        status_color = (30, 30, 30)

    text_lines = [
        f"status={status}  frame={frame_idx}  episode={episode_idx}  step={episode_step}",
        f"pos=({pos_w[0]:+.2f}, {pos_w[1]:+.2f}, {pos_w[2]:+.2f})  max_x={episode_max_x:.2f}",
        f"lin_vel=({lin_vel_w[0]:+.2f}, {lin_vel_w[1]:+.2f}, {lin_vel_w[2]:+.2f})  fwd={fwd_vel:+.2f} m/s",
        f"up_proj={up_proj:+.3f}  heading_proj={heading_proj:+.3f}  angle_to_tgt={angle_to_target:+.3f}",
        f"reward={reward:+.3f}  yaw_est={yaw_est:+.3f}  torso_z_obs={torso_z_obs:.3f}",
    ]

    line_h = 18
    pad_x = 12
    pad_y = 10
    telemetry_top = 12
    box = (
        20,
        telemetry_top,
        width - 20,
        telemetry_top + pad_y * 2 + line_h * len(text_lines) - 4,
    )
    draw.rectangle(box, fill=(252, 252, 253), outline=(188, 194, 204), width=1)
    ty = box[1] + pad_y
    for line in text_lines:
        draw.text((box[0] + pad_x, ty), line, fill=status_color)
        ty += line_h

    title_y = box[3] + 12
    panel_top = title_y + 24
    left_panel = (30, panel_top, width // 2 - 20, height - 40)
    right_panel = (width // 2 + 20, panel_top, width - 30, height - 40)

    draw.rectangle(left_panel, outline=(86, 94, 106), width=2)
    draw.rectangle(right_panel, outline=(86, 94, 106), width=2)
    draw.text((left_panel[0], title_y), "Side view (x-z)", fill=(30, 30, 30))
    draw.text((right_panel[0], title_y), "Top-down (x-y)", fill=(30, 30, 30))

    # --- Side panel (x-z): forward progress + height ---
    xz_pts = trail_xz[-60:] + [(float(pos_w[0]), float(pos_w[2]))]
    xz_bounds = _compute_bounds_2d(xz_pts, min_span=3.0, margin_ratio=0.2)
    # ensure ground reference is always visible
    xz_bounds = (
        xz_bounds[0],
        xz_bounds[1],
        min(xz_bounds[2], 0.0),
        max(xz_bounds[3], 2.5),
    )
    _draw_ground_line(draw, termination_height, xz_bounds, left_panel)
    _draw_trail(draw, trail_xz, xz_bounds, left_panel, (160, 210, 255))
    _draw_humanoid_side(draw, pos_w, dof_scaled, xz_bounds, left_panel, fallen)
    _draw_direction_arrow(draw, left_panel)

    # --- Top-down panel (x-y): lateral drift ---
    xy_pts = trail_xy[-60:] + [(float(pos_w[0]), float(pos_w[1]))]
    xy_bounds = _compute_bounds_2d(xy_pts, min_span=3.0, margin_ratio=0.2)
    _draw_trail(draw, trail_xy, xy_bounds, right_panel, (160, 210, 255))
    _draw_humanoid_topdown(draw, pos_w, dof_scaled, yaw_est, xy_bounds, right_panel, fallen)
    _draw_direction_arrow(draw, right_panel)

    return np.asarray(image, dtype=np.uint8)


# ---------------------------------------------------------------------------
# State extraction from env
# ---------------------------------------------------------------------------

def _read_humanoid_state(
    base_env,
    env_index: int,
    obs_row: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float, np.ndarray]:
    """Return (pos_w, lin_vel_w, ang_vel_w, heading_proj, up_proj, angle_to_target, torso_z_obs, dof_scaled)."""
    pos_w = None
    lin_vel_w = None
    ang_vel_w = None

    robot = getattr(base_env, "robot", None) or getattr(base_env, "_robot", None)
    if robot is not None and hasattr(robot, "data"):
        try:
            pos_w = robot.data.root_pos_w[env_index].detach().cpu().numpy()
            lin_vel_w = robot.data.root_lin_vel_w[env_index].detach().cpu().numpy()
            ang_vel_w = robot.data.root_ang_vel_w[env_index].detach().cpu().numpy()
        except Exception:
            pos_w = lin_vel_w = ang_vel_w = None

    # Observation-based fallbacks
    # obs[0]: torso_z  obs[1:4]: vel_loc  obs[4:7]: angvel_loc
    torso_z_obs = float(obs_row[0]) if obs_row.shape[0] > 0 else 1.0
    if pos_w is None:
        pos_w = np.array([0.0, 0.0, torso_z_obs], dtype=float)
    if lin_vel_w is None:
        lin_vel_w = np.array(obs_row[1:4], dtype=float) if obs_row.shape[0] >= 4 else np.zeros(3, dtype=float)
    if ang_vel_w is None:
        ang_vel_w = np.array(obs_row[4:7], dtype=float) if obs_row.shape[0] >= 7 else np.zeros(3, dtype=float)

    # obs[9]: angle_to_target  obs[10]: up_proj  obs[11]: heading_proj
    angle_to_target = float(obs_row[9])  if obs_row.shape[0] > 9  else 0.0
    up_proj         = float(obs_row[10]) if obs_row.shape[0] > 10 else 1.0
    heading_proj    = float(obs_row[11]) if obs_row.shape[0] > 11 else 1.0

    # obs[12:33]: dof_pos_scaled (21 joints, each in [-1, 1])
    dof_scaled = np.array(obs_row[12:33], dtype=float) if obs_row.shape[0] >= 33 else np.zeros(21, dtype=float)

    return pos_w, lin_vel_w, ang_vel_w, heading_proj, up_proj, angle_to_target, torso_z_obs, dof_scaled


# ---------------------------------------------------------------------------
# Main step-loop (shared between rsl_rl and rl_games)
# ---------------------------------------------------------------------------

def _run_loop(
    env,
    policy_fn,
    obs_getter,
    step_fn,
    simulation_app,
    args_cli,
    step_dt: float,
    termination_height: float,
    writer,
):
    env_index = max(0, args_cli.env_index)
    frame_idx = 0
    episode_idx = 0
    episode_step = 0
    yaw_est = 0.0
    trail_xz: list[tuple[float, float]] = []
    trail_xy: list[tuple[float, float]] = []
    episode_max_x = 0.0

    obs = obs_getter()

    while simulation_app.is_running():
        import torch

        with torch.inference_mode():
            actions, obs, rewards, dones = step_fn(policy_fn, obs)

        policy_obs = _extract_policy_obs(obs)
        if policy_obs.ndim == 1:
            policy_obs = policy_obs.unsqueeze(0)
        active_env_idx = min(env_index, policy_obs.shape[0] - 1)
        obs_row = policy_obs[active_env_idx].detach().cpu().numpy()

        pos_w, lin_vel_w, ang_vel_w, heading_proj, up_proj, angle_to_target, torso_z_obs, dof_scaled = (
            _read_humanoid_state(env.unwrapped, active_env_idx, obs_row)
        )

        reward = float(rewards[active_env_idx].item())
        done = bool(dones[active_env_idx].item())
        fallen = float(pos_w[2]) < termination_height

        yaw_est += float(ang_vel_w[2]) * step_dt
        episode_max_x = max(episode_max_x, float(pos_w[0]))

        trail_xz.append((float(pos_w[0]), float(pos_w[2])))
        trail_xy.append((float(pos_w[0]), float(pos_w[1])))
        if len(trail_xz) > args_cli.trail_length:
            trail_xz = trail_xz[-args_cli.trail_length :]
            trail_xy = trail_xy[-args_cli.trail_length :]

        frame = _render_humanoid_frame(
            pos_w=pos_w,
            lin_vel_w=lin_vel_w,
            ang_vel_w=ang_vel_w,
            heading_proj=heading_proj,
            up_proj=up_proj,
            angle_to_target=angle_to_target,
            torso_z_obs=torso_z_obs,
            dof_scaled=dof_scaled,
            reward=reward,
            done=done,
            fallen=fallen,
            frame_idx=frame_idx,
            episode_idx=episode_idx,
            episode_step=episode_step,
            yaw_est=yaw_est,
            termination_height=termination_height,
            trail_xz=trail_xz,
            trail_xy=trail_xy,
            episode_max_x=episode_max_x,
            width=args_cli.frame_width,
            height=args_cli.frame_height,
        )
        writer.append_data(frame)

        frame_idx += 1
        episode_step += 1
        if done:
            episode_idx += 1
            episode_step = 0
            episode_max_x = 0.0
            yaw_est = 0.0

        if frame_idx % 100 == 0:
            print(f"[INFO] Recorded {frame_idx} frames (episodes: {episode_idx})")

        if args_cli.max_frames > 0 and frame_idx >= args_cli.max_frames:
            break
        if args_cli.num_episodes > 0 and episode_idx >= args_cli.num_episodes:
            break

    return frame_idx, episode_idx


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args_cli):
    import math

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

    termination_height = float(getattr(env_cfg, "termination_height", 0.8))
    step_dt = float(getattr(env_cfg.sim, "dt", 1 / 120)) * float(getattr(env_cfg, "decimation", 2))

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.output_file)), exist_ok=True)
    writer = imageio.get_writer(args_cli.output_file, fps=args_cli.fps)

    print(f"[INFO] Device:             {device}")
    print(f"[INFO] Task:               {args_cli.task}")
    print(f"[INFO] Checkpoint:         {args_cli.checkpoint}")
    print(f"[INFO] Output video:       {args_cli.output_file}")
    print(f"[INFO] Num envs:           {env_cfg.scene.num_envs}")
    print(f"[INFO] Termination height: {termination_height}")
    print(f"[INFO] Step dt:            {step_dt:.4f} s")

    frame_idx = 0
    episode_idx = 0

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

            env_index = max(0, args_cli.env_index)
            frame_idx = 0
            episode_idx = 0
            episode_step = 0
            yaw_est = 0.0
            trail_xz: list[tuple[float, float]] = []
            trail_xy: list[tuple[float, float]] = []
            episode_max_x = 0.0

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

                pos_w, lin_vel_w, ang_vel_w, heading_proj, up_proj, angle_to_target, torso_z_obs, dof_scaled = (
                    _read_humanoid_state(env.unwrapped, active_env_idx, obs_row)
                )

                reward = float(rewards[active_env_idx].item())
                done = bool(dones[active_env_idx].item())
                fallen = float(pos_w[2]) < termination_height

                yaw_est += float(ang_vel_w[2]) * step_dt
                episode_max_x = max(episode_max_x, float(pos_w[0]))

                trail_xz.append((float(pos_w[0]), float(pos_w[2])))
                trail_xy.append((float(pos_w[0]), float(pos_w[1])))
                if len(trail_xz) > args_cli.trail_length:
                    trail_xz = trail_xz[-args_cli.trail_length :]
                    trail_xy = trail_xy[-args_cli.trail_length :]

                frame = _render_humanoid_frame(
                    pos_w=pos_w,
                    lin_vel_w=lin_vel_w,
                    ang_vel_w=ang_vel_w,
                    heading_proj=heading_proj,
                    up_proj=up_proj,
                    angle_to_target=angle_to_target,
                    torso_z_obs=torso_z_obs,
                    dof_scaled=dof_scaled,
                    reward=reward,
                    done=done,
                    fallen=fallen,
                    frame_idx=frame_idx,
                    episode_idx=episode_idx,
                    episode_step=episode_step,
                    yaw_est=yaw_est,
                    termination_height=termination_height,
                    trail_xz=trail_xz,
                    trail_xy=trail_xy,
                    episode_max_x=episode_max_x,
                    width=args_cli.frame_width,
                    height=args_cli.frame_height,
                )
                writer.append_data(frame)

                frame_idx += 1
                episode_step += 1
                if done:
                    episode_idx += 1
                    episode_step = 0
                    episode_max_x = 0.0
                    yaw_est = 0.0

                if frame_idx % 100 == 0:
                    print(f"[INFO] Recorded {frame_idx} frames (episodes: {episode_idx})")

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
            env_configurations.register(
                "rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env}
            )

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

            env_index = max(0, args_cli.env_index)
            episode_step = 0
            yaw_est = 0.0
            trail_xz: list[tuple[float, float]] = []
            trail_xy: list[tuple[float, float]] = []
            episode_max_x = 0.0

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

                pos_w, lin_vel_w, ang_vel_w, heading_proj, up_proj, angle_to_target, torso_z_obs, dof_scaled = (
                    _read_humanoid_state(env.unwrapped, active_env_idx, obs_row)
                )

                reward = float(rewards[active_env_idx].item())
                done = bool(dones[active_env_idx].item())
                fallen = float(pos_w[2]) < termination_height

                yaw_est += float(ang_vel_w[2]) * step_dt
                episode_max_x = max(episode_max_x, float(pos_w[0]))

                trail_xz.append((float(pos_w[0]), float(pos_w[2])))
                trail_xy.append((float(pos_w[0]), float(pos_w[1])))
                if len(trail_xz) > args_cli.trail_length:
                    trail_xz = trail_xz[-args_cli.trail_length :]
                    trail_xy = trail_xy[-args_cli.trail_length :]

                frame = _render_humanoid_frame(
                    pos_w=pos_w,
                    lin_vel_w=lin_vel_w,
                    ang_vel_w=ang_vel_w,
                    heading_proj=heading_proj,
                    up_proj=up_proj,
                    angle_to_target=angle_to_target,
                    torso_z_obs=torso_z_obs,
                    dof_scaled=dof_scaled,
                    reward=reward,
                    done=done,
                    fallen=fallen,
                    frame_idx=frame_idx,
                    episode_idx=episode_idx,
                    episode_step=episode_step,
                    yaw_est=yaw_est,
                    termination_height=termination_height,
                    trail_xz=trail_xz,
                    trail_xy=trail_xy,
                    episode_max_x=episode_max_x,
                    width=args_cli.frame_width,
                    height=args_cli.frame_height,
                )
                writer.append_data(frame)

                frame_idx += 1
                episode_step += 1
                if done:
                    episode_idx += 1
                    episode_step = 0
                    episode_max_x = 0.0
                    yaw_est = 0.0

                if frame_idx % 100 == 0:
                    print(f"[INFO] Recorded {frame_idx} frames (episodes: {episode_idx})")

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
        print(f"[INFO] Fallback recording complete. Frames: {frame_idx}, Episodes: {episode_idx}")
        print(f"[INFO] Saved video: {args_cli.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Record Humanoid rollout via observation/state fallback rendering."
    )
    parser.add_argument("--task", type=str, default="Isaac-Humanoid-Direct-v0", help="Name of the task.")
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
    parser.add_argument(
        "--output_file",
        type=str,
        default="./recordings/humanoid_fallback.mp4",
        help="Output MP4 path.",
    )
    parser.add_argument("--fps", type=int, default=30, help="Video FPS.")
    parser.add_argument(
        "--max_frames",
        type=int,
        default=1350,
        help="Maximum frames to write (1350 = ~15 s episode at 90 Hz / decimation=2 → 45 Hz → 30 fps). <=0 = unlimited.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help="Stop after this many completed episodes. <=0 = unlimited.",
    )
    parser.add_argument("--env_index", type=int, default=0, help="Environment index to visualize.")
    parser.add_argument(
        "--trail_length",
        type=int,
        default=200,
        help="How many recent positions to keep as trajectory trail.",
    )
    parser.add_argument("--frame_width", type=int, default=1280, help="Video frame width in pixels.")
    parser.add_argument("--frame_height", type=int, default=720, help="Video frame height in pixels.")
    args_cli = parser.parse_args()
    main(args_cli)
