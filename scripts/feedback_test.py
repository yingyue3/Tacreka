from isaaclab_eureka.managers.feedback_manager import (
    HumanFeedbackManager, 
    RewardInfo, 
    FeatureSpec
)

manager = HumanFeedbackManager(port=8889)

# Define detailed feature specs (like your JSON format)
feature_specs_v1 = [
    FeatureSpec(
        feature_name="position_error",
        intent="Encourage the agent to minimize the distance to the target position.",
        measurable_signals=["desired_pos_b"],
        proxy_metric="torch.norm(desired_pos_b)",
        weight=1.0,
        desired_direction="minimize",
        typical_failure_mode="The agent could hover in place near the target without considering smooth flight."
    ),
    FeatureSpec(
        feature_name="linear_velocity",
        intent="Encourage the agent to maintain a suitable forward speed toward the target.",
        measurable_signals=["root_lin_vel_b"],
        proxy_metric="torch.norm(root_lin_vel_b)",
        weight=0.5,
        desired_direction="maximize",
        typical_failure_mode="The agent could reach high speeds with abrupt movements, sacrificing smoothness."
    ),
    FeatureSpec(
        feature_name="angular_velocity",
        intent="Promote smooth rotational movements of the quadcopter for stable flight.",
        measurable_signals=["root_ang_vel_b"],
        proxy_metric="torch.norm(root_ang_vel_b)",
        weight=0.5,
        desired_direction="minimize",
        typical_failure_mode="The agent could move quickly in terms of angle but end up making jerky, unstable movements."
    ),
    FeatureSpec(
        feature_name="gravity_projection",
        intent="Ensure that the quadcopter maintains stable flight orientation relative to gravity.",
        measurable_signals=["projected_gravity_b"],
        proxy_metric="abs(projected_gravity_b[2])",
        weight=0.5,
        desired_direction="maximize",
        typical_failure_mode="The agent might ignore orientation and achieve projected gravity values only by tilting sharply."
    ),
]

feature_specs_v2 = [
    FeatureSpec(
        feature_name="distance_to_goal",
        intent="Minimize distance between quadcopter and target position.",
        measurable_signals=["desired_pos_b"],
        proxy_metric="1.0 / (1.0 + torch.norm(desired_pos_b))",
        weight=2.0,
        desired_direction="maximize",
        typical_failure_mode="May cause aggressive movements towards goal."
    ),
    FeatureSpec(
        feature_name="stability_bonus",
        intent="Reward stable hover with minimal oscillation.",
        measurable_signals=["root_ang_vel_b", "root_lin_vel_b"],
        proxy_metric="exp(-torch.norm(root_ang_vel_b) - 0.5 * torch.norm(root_lin_vel_b))",
        weight=1.0,
        desired_direction="maximize",
        typical_failure_mode="Could prioritize stability over reaching the goal."
    ),
]
feature_dic =[{"feature_name":"position_error","intent":"Encourage the agent to minimize the distance to the target position.","measurable_signals":["desired_pos_b"],"proxy_metric":"torch.norm(desired_pos_b)","weight":1.0,"desired_direction":"minimize","typical_failure_mode":"The agent could hover in place near the target without considering smooth flight."},{"feature_name":"linear_velocity","intent":"Encourage the agent to maintain a suitable forward speed toward the target.","measurable_signals":["root_lin_vel_b"],"proxy_metric":"torch.norm(root_lin_vel_b)","weight":0.5,"desired_direction":"maximize","typical_failure_mode":"The agent could reach high speeds with abrupt movements, sacrificing smoothness."},{"feature_name":"angular_velocity","intent":"Promote smooth rotational movements of the quadcopter for stable flight.","measurable_signals":["root_ang_vel_b"],"proxy_metric":"torch.norm(root_ang_vel_b)","weight":0.5,"desired_direction":"minimize","typical_failure_mode":"The agent could move quickly in terms of angle but end up making jerky, unstable movements."},{"feature_name":"gravity_projection","intent":"Ensure that the quadcopter maintains stable flight orientation relative to gravity.","measurable_signals":["projected_gravity_b"],"proxy_metric":"abs(projected_gravity_b[2])","weight":0.5,"desired_direction":"maximize","typical_failure_mode":"The agent might ignore orientation and achieve projected gravity values only by tilting sharply."}]

# Define reward info with feature specs
reward_info_v1 = RewardInfo(
    name="Feature-based Reward v1",
    description="Multi-objective reward combining position error, velocity control, and orientation stability.",
    feature_specs=feature_dic,
)

reward_info_v2 = RewardInfo(
    name="Simplified Reward v2",
    description="Simplified reward focusing on goal-reaching with stability bonus.",
    feature_specs=feature_specs_v2,
)

result = manager.select_video(
    video_paths=["NONE","./recordings/quad_tac_71.mp4"],
    descriptions=["Reward v1 (Multi-objective)", "Reward v2 (Simplified)"],
    task_description="Quadcopter hover task: The drone should reach and maintain a stable position at the target location while minimizing oscillations and energy consumption.",
    reward_infos=[reward_info_v1, reward_info_v2],
    allow_text_feedback=False,
    allow_rating=False,
)

print(f"\n=== Feedback Summary ===")
print(f"Human selected: {result.selected_video}")
print(f"Selected index: {result.selected_index}")
if result.text_feedback:
    print(f"Feedback: {result.text_feedback}")
if result.rating is not None:
    print(f"Rating: {result.rating}/10")
if result.selected_reward_info:
    print(f"Selected reward: {result.selected_reward_info.name}")
