"""Test script for the feature ranking + filtering interface."""

from isaaclab_eureka.managers.feedback_manager import HumanFeedbackManager

manager = HumanFeedbackManager(port=8889)

features = [
    {
        "feature_name": "position_error",
        "intent": "Encourage the agent to minimise the distance to the target position.",
        "measurable_signals": ["desired_pos_b"],
        "proxy_metric": "torch.norm(desired_pos_b, dim=-1)",
        "weight": 1.0,
        "desired_direction": "minimize",
        "typical_failure_mode": "Agent hovers in place near the target without smooth flight.",
    },
    {
        "feature_name": "linear_velocity",
        "intent": "Encourage a suitable forward speed toward the target.",
        "measurable_signals": ["root_lin_vel_b"],
        "proxy_metric": "torch.norm(root_lin_vel_b, dim=-1)",
        "weight": 0.5,
        "desired_direction": "maximize",
        "typical_failure_mode": "Agent may reach high speeds with abrupt movements.",
    },
    {
        "feature_name": "angular_velocity",
        "intent": "Promote smooth rotational movements for stable flight.",
        "measurable_signals": ["root_ang_vel_b"],
        "proxy_metric": "torch.norm(root_ang_vel_b, dim=-1)",
        "weight": 0.5,
        "desired_direction": "minimize",
        "typical_failure_mode": "Agent makes jerky, unstable angular movements.",
    },
    {
        "feature_name": "gravity_projection",
        "intent": "Maintain stable orientation relative to gravity.",
        "measurable_signals": ["projected_gravity_b"],
        "proxy_metric": "projected_gravity_b[:, 2]",
        "weight": 0.5,
        "desired_direction": "maximize",
        "typical_failure_mode": "Agent tilts sharply to satisfy the projection signal.",
    },
    {
        "feature_name": "action_smoothness",
        "intent": "Penalise rapid changes in motor commands.",
        "measurable_signals": ["actions", "prev_actions"],
        "proxy_metric": "torch.norm(actions - prev_actions, dim=-1)",
        "weight": 0.3,
        "desired_direction": "minimize",
        "typical_failure_mode": "Agent becomes too conservative and slow to react.",
    },
]

result = manager.rank_and_filter_features(
    feature_specs=features,
    task_description=(
        "Quadcopter hover task: rank the features by importance "
        "and drop any you consider unhelpful."
    ),
    allow_text_feedback=True,
    output_dir="./recordings",
)

print(f"\n{'=' * 55}")
print("FEATURE RANKING RESULTS")
print("=" * 55)

print(f"\nKept & ranked ({len(result.ranked_features)}):")
for i, feat in enumerate(result.ranked_features):
    name = feat.get("feature_name") if isinstance(feat, dict) else feat.feature_name
    print(f"  Rank {i + 1}: {name}")

print(f"\nDropped ({len(result.dropped_features)}):")
for feat in result.dropped_features:
    name = feat.get("feature_name") if isinstance(feat, dict) else feat.feature_name
    print(f"  - {name}")

if result.text_feedback:
    print(f"\nFeedback: {result.text_feedback}")

print(f"\nRanked indices : {result.ranked_indices}")
print(f"Dropped indices: {result.dropped_indices}")
