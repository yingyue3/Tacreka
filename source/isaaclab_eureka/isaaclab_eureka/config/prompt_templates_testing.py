# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

############### FEATURE GENERATION PROMPT TEMPLATES ###############

DIRECT_WORKFLOW_REWARD_FORMATTING_INSTRUCTIONS = """
Your reward function should use useful variables from the environment as inputs.
It must comply to the following signature exactly:

def _get_rewards_eureka(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ...
    return reward, individual_rewards_dict

Make sure any new tensor or variable you introduce is on the same device as self.device.
The output of the reward function should consist of two items:
    (1) the total reward, which has a dimension of (self.num_envs,) and is a torch.Tensor,
    (2) a dictionary of each individual reward component.
The code output should be formatted as a python code string: "```python ... ```" and contain only the get_rewards_eureka function.

Some helpful tips for writing the reward function code:
    (1) You may find it helpful to normalize the reward to a fixed range by applying transformations like torch.exp to the overall reward or its components
    (2) If you choose to transform a reward component, then you must also introduce a temperature parameter inside the transformation function; this parameter must be a named variable in the reward function and it must not be an input variable. Each transformed reward component should have its own temperature variable
    (3) Make sure the type of each input variable is correctly specified; a float input variable should not be specified as torch.Tensor
    (4) Most importantly, the reward code's input variables must contain only attributes of the provided environment class definition (namely, variables that have prefix self.). Under no circumstance can you introduce new input variables.
"""

TEST_FEATURE_GEN_FORMATTING_PROMPT = """
Instructions:
1) Propose 1 to 4 candidate features. Each feature must be:
   - Interpretable to a non-expert human (1 sentence description).
   - Measurable from the given observations/actions (no hidden variables).
   - Focused on behavior/outcomes (not "learn faster" or "high return").
   - As independent as possible.
   - Based on a signal that varies MEANINGFULLY across typical agent states — not a near-constant.

2) For each feature, output the following fields:
   - feature_name: short identifier
   - intent: what this feature encourages (1 sentence)
   - desired_direction: maximize or minimize
   - typical_failure_mode: how an agent could "game" this feature if it were rewarded alone

Output format: valid JSON ONLY. Do not include markdown, code fences, comments, or extra text.
"""

TEST_FEATURE_GEN_FEEDBACK_PROMPT = """
We trained a RL policy using the reward function generated from the provided reward feature decomposition and tracked the values of the individual components in the reward function as well as global policy metrics such as success rates and episode lengths after every {feedback_subsampling} epochs and the maximum, mean, minimum values encountered:
"""

TEST_FEATURE_GEN_INITIAL_PROMPT = """
You are a reward-design assistant for reinforcement learning.

Goal: Decompose the following RL task into a small set of interpretable "features" that capture what humans would consider good performance. These features will later be turned into reward terms and combined as a weighted sum.
""" + TEST_FEATURE_GEN_FORMATTING_PROMPT

TEST_FEATURE_GEN_EXPLORE_FEEDBACK_PROMPT = """
Please carefully analyze the policy feedback and provide a new reward feature decomposition that sets the training results as close as possible to the desired task score. The new reward feature decomposition should have at least one brand new feature.
Some helpful tips for analyzing the policy feedback:
    (1) If the success rates are always near zero, then you must add a new reward feature decomposition component or consider dropping existing ones.
    (2) If the values for a certain reward component are near identical throughout (min ≈ max, range < 0.05),
        this component is measuring a near-constant signal and provides NO gradient to the policy. You MUST:
        (a) Discard the reward feature entirely and replace with one that has a non-trivial signal_range
    (3) Do NOT reference attributes that are not explicitly provided in the environment's observation method.
Please analyze each existing reward component in the suggested manner above first, and then write the reward feature decomposition.
""" + TEST_FEATURE_GEN_FORMATTING_PROMPT

TEST_FEATURE_GEN_EXPLOIT_FEEDBACK_PROMPT = """
Please carefully analyze the policy feedback and provide a new reward feature decomposition that sets the training results as close as possible to the desired task score. The new reward feature decomposition should have at least one feature with changed weights or discard one of the existing features.
Some helpful tips for analyzing the policy feedback:
    (1) If the success rates are always near zero, then consider dropping unhelpful reward feature components.
    (2) If the values for a certain reward component are near identical throughout (range < 0.05), this component
        is measuring a near-constant signal. You may consider:
        (a) Replace it with a directional component or a signal that genuinely varies with agent behavior.
            For example, replace torch.norm(projected_gravity_b) with projected_gravity_b[:, 2] for tilt.
        (b) Discarding the reward feature entirely.
    (3) If some reward feature magnitudes are significantly larger than others, re-scale to a comparable range.
    (4) If the total reward grew more than 5x during training, the features are not properly bounded;
        add normalization constraints (e.g., torch.tanh or torch.clamp) in the implementation.
Please analyze each existing reward feature in the suggested manner above first, and then write the reward feature decomposition.
""" + TEST_FEATURE_GEN_FORMATTING_PROMPT

# FEATURE_GEN_PROMPT = """
# Decompose the following RL task into a small set of interpretable "features" that capture what humans would consider good performance. These features will later be turned into reward terms and combined as a weighted sum.
# Task context:
# - Task description is: {task_description}
# - The desired task score is: {success_metric_to_win}
# - Here is how we get the observations from the environment: {get_observations_method_as_string}
# """

TEST_FEATURE_GEN_PROMPT = """
Decompose the following RL task into a small set of interpretable "features" that capture what humans would consider good performance. These features will later be turned into reward terms and combined as a weighted sum.
Task context:
- Task description is: {task_description}
- The desired task score is: {success_metric_to_win}
- Here is how we get the observations from the environment: {get_observations_method_as_string}
"""

############### Single reward function generation templates ###############

TEST_FEATURE_AS_ONE_REWARD_FORMATTING_PROMPT = """
Your reward function should use useful variables from the environment as inputs.
It must comply to the following signature exactly:

def _get_rewards_eureka(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ...
    return reward, individual_rewards_dict

Make sure any new tensor or variable you introduce is on the same device as self.device.
The output of the reward function should consist of two items:
    (1) the total reward, which has a dimension of (self.num_envs,) and is a torch.Tensor,
    (2) a dictionary of each individual reward component.
The code output should be formatted as a python code string: "```python ... ```" and contain only the get_rewards_eureka function.

Some helpful tips for writing the reward function code:
    (1) Normalize each component to a fixed range such as [0, 1] using transformations like torch.exp(-t * signal)
        or torch.tanh. This prevents reward explosion as training progresses.
    (2) Choose temperature parameters carefully: for torch.exp(-t * signal), set t so that
        t * (typical_signal_magnitude) is in the range [0.5, 3.0]. If position error is typically 1–5 m,
        use t ≈ 0.2–1.0. Each transformed component must have its own named temperature variable.
    (3) Make sure the type of each input variable is correctly specified; a float input variable should not be
        specified as torch.Tensor.
    (4) Most importantly, the reward code's input variables must contain only attributes of the provided
        environment class definition (namely, variables with prefix self.). Do NOT introduce new input variables
        or reference attributes not explicitly shown in the observation method.

Critical pitfalls to avoid (each has caused zero-signal or diverging components in past runs):
    (P1) Do NOT use torch.norm(projected_gravity_b, dim=-1) as a stability or uprightness signal.
         Its magnitude is approximately constant at 9.81 m/s^2 regardless of robot orientation, so it
         provides zero gradient signal. Instead use projected_gravity_b[:, 2] (body-frame z-component):
         it is near -9.81 when level and deviates toward 0 when the robot is severely tilted.
    (P2) When computing position error in body frame, use subtract_frame_transforms(root_pos_w, root_quat_w,
         desired_pos_w) rather than simple vector subtraction, to correctly account for orientation.
    (P3) Do NOT reference attributes that do not exist in the environment class (e.g., do not assume
         previous_root_lin_vel_b exists unless it is shown in the observation method).
    (P4) After applying torch.exp(-t * signal), verify the output range is meaningful. If signal has low
         variance (e.g., near-constant), the component will also be near-constant. Choose a different signal.

Hard requirements:
1) Implement one reward component per feature: r_(feature_name) as a Tensor of shape (num_envs,).
2) Normalize / bound each component to roughly [-1, 1] or [0, 1] (use exp, tanh, or smooth saturation).
3) Avoid sparse-only rewards: each component should provide a learning signal across most states.
4) Mitigate reward hacking: for each feature, add a small safeguard term if needed to prevent its "typical_failure_mode".
5) Use consistent scaling so no single component dominates by accident.
6) Provide a final reward:
   R = sum_i w_i * r_i
   Use default weights provided in the FEATURES_JSON unless you have a strong reason; if you change weights, explain why.
7) The reward code's input variables must contain only attributes of the provided environment class (prefix self.).
8) The final combined reward must be positively correlated with task success: improving R should correspond
   to the agent performing better on the stated task objective.

Output requirements:
- Output ONLY a single Python code block.
- The code must define exactly one function:
"""

TEST_FEATURE_AS_ONE_REWARD_INITIAL_PROMPT = """
You are a reward engineer for reinforcement learning.
Goal: Write reward functions for an IsaacLab task by turning a given set of human-interpretable features into reward components, then composing them into a final reward as a weighted sum.
""" + TEST_FEATURE_AS_ONE_REWARD_FORMATTING_PROMPT

# FEATURE_AS_ONE_REWARD_PROMPT = """
# Write a reward function for the following task: {task_description}
# The desired task score is: {success_metric_to_win}
# Here is how we get the observations from the environment: {get_observations_method_as_string}

# Features to implement (generated previously):
# {FEATURES_JSON}
# """

TEST_FEATURE_AS_ONE_REWARD_PROMPT = """
Write a reward function for the following task: {task_description}
Here is how we get the observations from the environment: {get_observations_method_as_string}

Features to implement (generated previously):
{FEATURES_JSON}
"""

'''
Each feature contains:
- feature_name
- intent
- measurable_signals (names from SIGNALS_JSON)
- proxy_metric (plain text formula)
- signal_range (estimated [min, max] under typical behavior)
- desired_direction (maximize/minimize)
- typical_failure_mode
'''

TEST_FEATURE_AS_ONE_FAILURE_FEEDBACK_PROMPT = """
Executing the reward function code above has the following error: {traceback_msg}.
Please fix the bug and provide a new, improved reward function!
""" + TEST_FEATURE_AS_ONE_REWARD_FORMATTING_PROMPT

TEST_FEATURE_AS_ONE_SUCCESS_PRE_FEEDBACK_PROMPT = """
We trained a RL policy using the provided reward function code and tracked the values of the individual components in the reward function as well as global policy metrics such as success rates and episode lengths after every {feedback_subsampling} epochs and the maximum, mean, minimum values encountered:
"""

TEST_FEATURE_AS_ONE_SUCCESS_POST_FEEDBACK_PROMPT = """
Please carefully analyze the policy feedback and provide a new, improved reward function that can better solve the task. Some helpful tips for analyzing the policy feedback:
    (1) If the success rates are always near zero, then you must rewrite the entire reward function.
    (2) If the values for a certain reward component are near identical throughout (Min ≈ Max, range < 0.05),
        this component is measuring a near-constant signal and provides NO gradient to the policy. You MUST:
        (a) Identify the root cause — e.g., using torch.norm() of a fixed-magnitude vector like projected_gravity_b.
        (b) Replace it with a genuinely varying signal (e.g., projected_gravity_b[:, 2] for tilt/uprightness).
        (c) Alternatively, re-write or discard the component entirely.
    (3) If some reward components' magnitude is significantly larger, then you must re-scale to a proper range.
    (4) If the total reward magnitude grew more than 5x during training (e.g., from ~1 to ~40), the components
        are not properly bounded. Apply torch.tanh or torch.clamp to keep each component in [-1, 1] or [0, 1].
    (5) If the task_score improves initially but then degrades (reward-objective misalignment), strengthen
        the primary_success component weight and/or remove auxiliary terms that pull in a conflicting direction.
Please analyze each existing reward component in the suggested manner above first, and then write the reward function code based on the refined set of features.
Features to implement (refined): {FEATURES_JSON}
""" + DIRECT_WORKFLOW_REWARD_FORMATTING_INSTRUCTIONS


############### Per-component refinement (locked feature set) ###############

# System prompt used when refining a SINGLE feature in the locked set.
TEST_PER_COMPONENT_REFINEMENT_SYSTEM_PROMPT = """
You are a reward-feature refinement assistant for reinforcement learning.

Your job is to refine ONE feature of an already-locked reward feature set, based on
the per-component statistics observed during the most recent training run. The set of
feature_names is FROZEN — you must NEVER add features, remove features, rename features,
merge features, or split features. You only revise the *description* of the single
feature you are given so that the next reward implementation is more effective.
"""

TEST_PER_COMPONENT_REFINEMENT_PROMPT = """
We are evolving a reward function for an Isaac Lab RL task. The reward feature set was
locked after the initial exploration phase. We are now refining EACH feature individually,
one at a time, based on how it behaved during the latest training run.

Task: {task_description}
The desired task_score to reach is: {success_metric_to_win}.
Observation method (do NOT invent new attributes):
{get_observations_method_as_string}

You are refining ONE feature in the locked set. The other features are refined separately.
You MUST keep `feature_name` exactly as given. You MAY revise:
  - intent
  - desired_direction
  - typical_failure_mode

The feature being refined (from the locked set):
{feature_json}

For context, the full locked set of feature names is:
{locked_feature_names}

Component-level feedback for THIS component during the last run
(sub-sampled trajectory, then min/max/mean):
{component_feedback}

Overall task feedback for the last run (task_score trajectory and aggregate metrics):
{task_feedback}

Refinement guidance — go through each rule before producing your output:
  (1) NEAR-CONSTANT SIGNAL. If this component's values are nearly constant (Min ≈ Max,
      range < 0.05), the underlying signal carries no learning gradient. Rewrite `intent`
      so the next reward implementation will pick a more variable, behaviour-correlated
      signal (e.g., directional projections instead of vector norms).
  (2) DOMINATING MAGNITUDE. If this component's Max is more than 5x larger than the other
      components' Maxes (visible in the task feedback), update `typical_failure_mode` to
      flag the dominance and adjust `intent` so the next implementation uses tighter
      normalisation (tanh / clamp / smaller temperature).
  (3) MISALIGNED WITH TASK. If this component grew while task_score moved AWAY from
      {success_metric_to_win}, the component is likely anti-correlated with task success.
      Flip `desired_direction` or rewrite `intent` so the next implementation penalises
      the corresponding behaviour instead of rewarding it.
  (4) WELL-BEHAVED. If this component is clearly aligned with task progress and varies
      meaningfully, you may keep it almost the same. Returning identical text is allowed
      ONLY when the component is clearly aligned with task progress.
  (5) DO NOT add or remove components. DO NOT rename. Refine THIS feature only.

Output format: VALID JSON ONLY (no markdown, no code fences, no comments).
Schema:
{{
  "feature_name": "...",        // MUST equal the feature_name above
  "intent": "...",
  "desired_direction": "...",
  "typical_failure_mode": "..."
}}
"""


############### Reward generation from a refined LOCKED feature set ###############

TEST_LOCKED_FEATURE_REWARD_PROMPT = """
Write a reward function for the following task: {task_description}
Here is how we get the observations from the environment: {get_observations_method_as_string}

The reward function MUST implement EXACTLY the following LOCKED set of features.
Do NOT add, remove, rename, or merge any feature. Implement one reward component per
feature. Use the same feature_name as the dict key in `individual_rewards_dict` so the
training feedback can be parsed back per-component.

Locked + refined features (this iteration):
{FEATURES_JSON}

Each feature in the locked set was refined INDIVIDUALLY based on its observed signal in
the previous training run. Use each feature's refined `intent`, `desired_direction`, and
`typical_failure_mode` to choose the signal, transformation, and scale for that component.

Previous reward function code (for context — improve it, but keep the same locked
feature_names):
```python
{PREVIOUS_REWARD_CODE}
```

Last run's task feedback (for context):
{task_feedback}
""" + TEST_FEATURE_AS_ONE_REWARD_FORMATTING_PROMPT