# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

TASKS_CFG = {
    "Isaac-Cartpole-Direct-v0": {
        "description": "balance a pole on a cart so that the pole stays upright",
        "success_metric": "self.episode_length_buf[env_ids].float().mean() / self.max_episode_length",
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.01,
    },
    "Isaac-Quadcopter-Direct-v0": {
        "description": (
            "bring the quadcopter to the target position: self._desired_pos_w, while making sure it flies smoothly"
        ),
        "success_metric": (
            "torch.linalg.norm(self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1).mean()"
        ),
        "success_metric_to_win": 0.0,
        "success_metric_tolerance": 0.2,
    },
    "Isaac-Reach-Franka-v0": {
        "description": (
            "Move the end-effector to a sampled target pose with the Franka robot"
        ),
        "success_metric": (
            "torch.linalg.norm("
            "combine_frame_transforms("
            "self.scene['robot'].data.root_pos_w[env_ids],"
            " self.scene['robot'].data.root_quat_w[env_ids],"
            " self.command_manager.get_command('ee_pose')[env_ids, :3]"
            ")[0]"
            " - self.scene['robot'].data.body_pos_w["
            "env_ids, self.scene['robot'].body_names.index('panda_hand')"
            "], dim=1).mean()"
        ),
        "success_metric_to_win": 0.0,
        "success_metric_tolerance": 0.05,
    },
    "Isaac-Lift-Cube-Franka-v0": {
        "description": (
            "Control a 7-DOF Franka Panda arm with a binary gripper to lift a cube off a table and "
            "place it at a randomized 3-D goal position. "
            "The cube's resting position is approximately (0.5, ±0.25, 0.055) m in the robot-base frame "
            "(X=0.5 m in front of robot, Y randomized ±0.25 m, Z=0.055 m — just above the table). "
            "The commanded goal is self.command_manager.get_command('object_pose')[:, :3] in robot-base "
            "frame: X ∈ (0.4, 0.6), Y ∈ (-0.25, 0.25), Z ∈ (0.25, 0.5) m — always well above the table. "
            "The gripper joint names are 'panda_finger_joint1' and 'panda_finger_joint2' (last 2 entries "
            "of joint_pos); close_command=0.0, open_command=0.04. "
            "CRITICAL — a successful reward function MUST include two stages: "
            "(1) a reaching reward (EE → cube distance) so the gripper gets close enough to grasp; "
            "(2) a goal-tracking reward (cube → goal distance) that activates once the cube is grasped "
            "and lifted. Without the reaching stage the cube never moves and the metric flatlines. "
            "Key state: self.scene['object'].data.root_pos_w (cube world pos), "
            "self.scene['robot'].data.root_pos_w (robot base world pos, identity rotation), "
            "self.scene['ee_frame'].data.target_pos_w[..., 0, :] (end-effector world pos)."
        ),
        # Episode-end mean distance (m) from cube to commanded goal. Lower is
        # better. Robot base uses identity rotation, so the world-frame goal is
        # just base position + commanded (base-frame) offset.
        "success_metric": (
            "torch.linalg.norm("
            "(self.scene['robot'].data.root_pos_w[env_ids]"
            " + self.command_manager.get_command('object_pose')[env_ids, :3])"
            " - self.scene['object'].data.root_pos_w[env_ids, :3], dim=1).mean()"
        ),
        "success_metric_to_win": 0.0,
        "success_metric_tolerance": 0.05,
    },
    "Isaac-Franka-Cabinet-Direct-v0": {
        "description": (
            "Control a 9-DOF Franka Panda arm (7 arm joints + 2 gripper fingers, action space 9, "
            "observation space 23) to open the cabinet drawer. "
            "The task is solved when the cabinet joint self._cabinet.data.joint_pos[:, 3] exceeds "
            "0.39 m. The joint starts at 0.0 on every reset and only moves if the gripper actually "
            "pulls the handle. "
            "The episode TERMINATES the instant the drawer passes 0.39 m; otherwise it runs the "
            "full 500 steps. A reward that pays a positive amount per step while the drawer is shut "
            "therefore pays the agent to keep it shut, because opening forfeits all remaining "
            "per-step reward. "
            "USE PROGRESS (DIFFERENCE) REWARDS as the primary mechanism: reward the per-step CHANGE "
            "in each quantity, not its level. A progress reward is automatically zero while nothing "
            "moves, so idling earns nothing and there is no incentive to stall. "
            "Required structure, with the direction of every term stated explicitly: "
            "(1) reach — reward must be POSITIVE when the distance from self.robot_grasp_pos to "
            "self.drawer_grasp_pos DECREASES: r_reach = (prev_dist - dist). "
            "(2) align — reward must be POSITIVE when the gripper-to-handle orientation error "
            "DECREASES: r_align = (prev_err - err). "
            "(3) pull — reward must be POSITIVE when self._cabinet.data.joint_pos[:, 3] INCREASES: "
            "r_pull = (drawer_pos - prev_drawer_pos), with the largest weight of the three. "
            "(4) success — a single terminal bonus, r_success = torch.where(drawer_pos > 0.39, B, 0), "
            "with B >= 10.0. Episode-summed progress rewards telescope to at most ~0.5, so B must be "
            "far larger than that to make opening clearly worth it. "
            "To keep previous values, lazily create the buffer and RE-ZERO the entries of the "
            "environments that were just reset, e.g. store self._prev_dist and refresh it at the end "
            "of the reward function; if you cannot detect resets, clamp each difference with "
            "torch.clamp(delta, min=-C) so the drawer snapping from 0.39 back to 0.0 at reset cannot "
            "produce a large spurious negative spike. "
            "A simpler alternative is allowed: make reach and align plain NEGATIVE penalties that are "
            "0 at the goal and negative away from it (r = -dist, r = -err), and make pull the "
            "positive continuous term drawer_pos / 0.39. This also removes the idling income. "
            "FORBIDDEN, because each of these has already been tried and failed: "
            "(a) any term of the form 1 - exp(-t * error) or -exp(-t * error) for reach or align — "
            "both are maximized by moving AWAY from the handle and mis-aligning the gripper; "
            "(b) conditioning the pull term on drawer_pos > 0.39 or clamp(drawer_pos - 0.39) — the "
            "episode ends at that threshold, so such a term can pay at most one step per episode and "
            "measures zero throughout training; use the continuous position or its change instead; "
            "(c) masking or zeroing the total reward from the state, such as "
            "total_reward[drawer_pos == 0.0] = 0.0 or total_reward * (1 - (drawer_pos <= 0)) — the "
            "drawer is at 0.0 on almost every step, so this deletes the entire learning signal; "
            "(d) a negative weight on a component that is already maximal at the goal, and a negative "
            "weight on an already-negative penalty (two negatives pay the agent to flail). "
            "Every component must increase as the task is performed better. "
            "Scale note: the handle distance is roughly 0.1-0.4 m, so if you do transform a level with "
            "exp(-t * distance) then t must be about 5-15; t near 0.2 leaves the term pinned at its "
            "maximum and it degenerates into a constant. "
            "Key state: self.robot_grasp_pos / self.robot_grasp_rot (gripper grasp frame), "
            "self.drawer_grasp_pos / self.drawer_grasp_rot (handle grasp frame), "
            "self._robot.data.body_pos_w[:, self.left_finger_link_idx] and "
            "[:, self.right_finger_link_idx] (finger world positions), "
            "self._robot.data.joint_pos / joint_vel, self._cabinet.data.joint_pos, self.actions."
        ),
        # Mean normalized drawer progress at episode end. This remains continuous
        # below the 0.39 m success threshold and is capped at 1.0 so opening past
        # the threshold can never make a candidate score worse. Evaluated before
        # _reset_idx_original zeroes the cabinet joints.
        "success_metric": (
            "torch.clamp(self._cabinet.data.joint_pos[env_ids, 3] / 0.39, min=0.0, max=1.0).mean()"
        ),
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.05,
    },
    "Isaac-Humanoid-Direct-v0": {
        "description": (
            "To make the humanoid run as fast as possible. "
            "A 21-DOF MuJoCo-style humanoid (action space 21, observation space 75) is driven by joint "
            "efforts and must run along the world +X axis, towards a target 1000 m away that it can "
            "never reach, so the task is really 'sustain the highest forward speed you can'. "
            "The episode lasts 15 s (self.max_episode_length_s) at a 60 Hz control rate, and terminates "
            "early the moment the torso drops below 0.8 m (self.cfg.termination_height), so falling "
            "forfeits every remaining step. Speed that cannot be held upright is therefore worth little: "
            "a reward must pay for forward progress AND for staying on its feet. "
            "Key state, all refreshed in _compute_intermediate_values before the reward is computed: "
            "self.torso_position / self.torso_rotation (root world pose; index 2 of the position is the "
            "torso height used by the termination check), self.velocity / self.ang_velocity (root world "
            "linear and angular velocity, index 0 of the linear velocity being the forward component), "
            "self.vel_loc / self.angvel_loc (the same velocities in the torso frame), "
            "self.up_proj (1.0 when the torso is perfectly upright), self.heading_proj (1.0 when the "
            "torso faces the target), self.angle_to_target, self.roll / self.pitch / self.yaw, "
            "self.dof_pos / self.dof_vel and self.dof_pos_scaled (joint positions normalized to "
            "[-1, 1] against the soft limits, so magnitudes near 1 mean a joint is at its limit), "
            "self.potentials / self.prev_potentials (negative distance to the target divided by the "
            "physics dt, so their difference is a per-step progress term), self.actions, and "
            "self.reset_terminated (True for the environments that fell this step)."
        ),
        # Mean forward speed sustained over the episode: the distance covered along world +X since the
        # start of the episode, divided by the full 15 s episode budget. Environments start at their
        # env origin (HUMANOID_CFG places the robot at x = 0 within the env), so the displacement is
        # just the torso's x offset from that origin, and this is read before _reset_idx_original
        # teleports the robot back.
        #
        # Deliberately divided by the whole episode budget rather than by the time actually survived,
        # and read from self.robot.data rather than the self.torso_position cached in
        # _compute_intermediate_values, for two reasons: dividing by the elapsed time would score a
        # humanoid that dives forward and falls after one second as highly as one that runs the full
        # 15 s, and self.robot.data is populated from the moment the scene is created, whereas the
        # cached values only exist once _compute_intermediate_values has run.
        "success_metric": (
            "((self.robot.data.root_pos_w[env_ids, 0] - self.scene.env_origins[env_ids, 0])"
            " / self.max_episode_length_s).mean()"
        ),
        # A well-trained humanoid holds roughly 5 m/s, i.e. about 75 m over the 15 s episode.
        "success_metric_to_win": 5.0,
        "success_metric_tolerance": 0.5,
    },
}
"""Configuration for the tasks supported by Isaac Lab Eureka.

`TASKS_CFG` is a dictionary that maps task names to their configuration. Each task configuration
is a dictionary that contains the following keys:

- `description`: A description of the task.
- `success_metric`: A Python expression that computes the success metric for the task.
- `success_metric_to_win`: The threshold for the success metric to win the task and stop.
- `success_metric_tolerance`: The tolerance for the success metric to consider the task successful.
"""
