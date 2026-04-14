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
            "lift a cube off a table with a Franka Panda arm and place it at a randomized goal position "
            "commanded by self.command_manager.get_command('object_pose') (first 3 components are the "
            "desired cube position in the robot root frame; goal z is between 0.25 and 0.5 m above the base). "
            "Key state: self.scene['object'].data.root_pos_w (cube world pos), "
            "self.scene['robot'].data.root_pos_w (robot base world pos, identity rotation), "
            "self.scene['ee_frame'].data.target_pos_w[..., 0, :] (end-effector world pos)."
        ),
        # Mean L2 distance from the cube to the commanded goal position at episode end.
        # The robot base has identity rotation (FRANKA_PANDA_CFG default), so the goal in world frame
        # equals robot_root_pos_w + des_pos_b (no quaternion rotation needed).
        # Lower is better; a perfect agent drives this to 0.
        "success_metric": (
            "torch.linalg.norm("
            "(self.scene['robot'].data.root_pos_w[env_ids]"
            " + self.command_manager.get_command('object_pose')[env_ids, :3])"
            " - self.scene['object'].data.root_pos_w[env_ids, :3],"
            " dim=1).mean()"
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
