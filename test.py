def _get_rewards_eureka(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Temperature parameters for normalization    
    temp_pole_uprightness = 0.01
    temp_cart_stability = 0.01
    temp_pole_velocity = 0.01
    temp_cart_velocity = 0.01
    temp_cart_position = 0.01
    temp_pole_position = 0.01
    
    # Reward Component Calculations
    r_pole_uprightness = torch.exp(-temp_pole_uprightness * torch.abs(90.0 - self.joint_pos[:, self._pole_dof_idx[0]]))
    r_cart_stability = torch.exp(-temp_cart_stability * torch.abs(self.joint_pos[:, self._cart_dof_idx[0]] - self.previous_joint_pos))
    r_pole_velocity = torch.exp(-temp_pole_velocity * torch.abs(self.joint_vel[:, self._pole_dof_idx[0]]))
    r_cart_velocity = torch.exp(-temp_cart_velocity * torch.abs(self.joint_vel[:, self._cart_dof_idx[0]]))
    r_cart_position = torch.exp(-temp_cart_position * torch.abs(self.max_position_bound - self.joint_pos[:, self._cart_dof_idx[0]]))
    r_pole_position = torch.exp(-temp_pole_position * torch.abs(90.0 - self.joint_pos[:, self._pole_dof_idx[0]]))

    # Safeguard Terms
    safeguard_pole_uprightness = 0.1 * (self.joint_pos[:, self._pole_dof_idx[0]] != 0.0).float() # Encourage agent to move the pole
    safeguard_cart_stability = 0.1 * (self.joint_pos[:, self._cart_dof_idx[0]] != 0.0).float() # Encourage agent to move the cart
    safeguard_pole_velocity = 0.1 * (self.joint_vel[:, self._pole_dof_idx[0]] != 0.0).float() # Encourage agent to change pole's velocity
    safeguard_cart_velocity = 0.1 * (self.joint_vel[:, self._cart_dof_idx[0]] != 0.0).float() # Encourage agent to change cart's velocity
    safeguard_cart_position = 0.1 * (self.joint_pos[:, self._cart_dof_idx[0]] != self.max_position_bound).float() # Encourage agent to move the cart away from the boundaries
    safeguard_pole_position = 0.1 * (self.joint_pos[:, self._pole_dof_idx[0]] != 90.0).float() # Encourage agent to adjust pole's position
    
    # Update Reward Components with Safeguards
    r_pole_uprightness += safeguard_pole_uprightness
    r_cart_stability += safeguard_cart_stability
    r_pole_velocity += safeguard_pole_velocity
    r_cart_velocity += safeguard_cart_velocity
    r_cart_position += safeguard_cart_position
    r_pole_position += safeguard_pole_position
    
    # Final Reward Calculation
    reward = r_pole_uprightness + r_cart_stability + r_pole_velocity + r_cart_velocity + r_cart_position + r_pole_position

    # Individual Rewards Dictionary
    individual_rewards_dict = {
            'pole_uprightness': r_pole_uprightness,
            'cart_stability': r_cart_stability,
            'pole_velocity': r_pole_velocity,
            'cart_velocity': r_cart_velocity,
            'cart_position': r_cart_position,
            'pole_position': r_pole_position,
            }
    
    return reward.to(self.device), individual_rewards_dict