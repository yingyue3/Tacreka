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
        # Mean L2 distance from the cube to the commanded goal position at episode end.
        # Uses combine_frame_transforms for correctness under any base orientation.
        # Lower is better; a perfect agent drives this to 0.
        "success_metric": (
            "combine_frame_transforms("
            "self.scene['robot'].data.root_pos_w[env_ids],"
            " self.scene['robot'].data.root_quat_w[env_ids],"
            " self.command_manager.get_command('object_pose')[env_ids, :3])[0]"
            ".sub(self.scene['object'].data.root_pos_w[env_ids, :3])"
            ".norm(dim=1).mean()"
        ),
        "success_metric_to_win": 0.0,
        # A well-trained agent achieves ~2–5 cm mean object-to-goal distance; accept ≤5 cm as solved.
        "success_metric_tolerance": 0.05,
    },
    "Isaac-Humanoid-Direct-v0": {
        "description": (
            "move the MuJoCo humanoid robot forward in the positive x direction as fast as possible "
            "while staying upright. The robot must not fall over (torso height must stay above "
            # "self.cfg.termination_height). The target direction is self.targets, which is fixed far "
            # "along the positive x axis. Key state variables available: self.torso_position (root position), "
            # "self.velocity (root linear velocity in world frame), self.heading_proj (projection of robot "
            # "heading onto target direction, in [-1, 1]), self.up_proj (projection of robot up-axis onto "
            # "world up-axis, in [-1, 1]), self.angle_to_target (signed angle between forward and target), "
            # "self.potentials (negative distance to target / dt), self.dof_pos, self.dof_vel, self.actions."
        ),
        # Forward velocity of the torso averaged over the resetting environments.
        # self.velocity is root_lin_vel_w, index 0 is the world-x (forward) component.
        # This is computed in _compute_intermediate_values which is called inside _get_dones
        # just before _reset_idx, so the value is current at reset time.
        "success_metric": "self.robot.data.root_lin_vel_w[env_ids, 0].mean()",
        # A well-trained MuJoCo humanoid reaches roughly 5 m/s forward speed.
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
