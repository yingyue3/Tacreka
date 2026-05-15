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

FEATURE_GEN_FORMATTING_PROMPT = """
Instructions:
1) Propose 1 to 4 candidate features. Each feature must be:
   - Interpretable to a non-expert human (1 sentence description).
   - Measurable from the given observations/actions (no hidden variables).
   - Focused on behavior/outcomes (not "learn faster" or "high return").
   - As independent as possible (avoid duplicates like "stability" and "uprightness" unless you clearly distinguish them).
   - Based on a signal that varies MEANINGFULLY across typical agent states — not a near-constant.

2) For each feature, output the following fields:
   - feature_name: short identifier
   - intent: what this feature encourages (1 sentence)
   - measurable_signals: which observation/action variables to use (explicit names from OBS_LIST / ACT_LIST)
   - proxy_metric: a concrete scalar metric formula in plain text (e.g., “abs(pole_angle)”, “-||cart_pos||”, “exp(-k*abs(angle))”)
   - weight: the weight of the feature in the final reward function
   - desired_direction: maximize or minimize
   - typical_failure_mode: how an agent could "game" this feature if it were rewarded alone

Some helpful tips for selecting the measurable signals:
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
Output format: valid JSON ONLY. Do not include markdown, code fences, comments, or extra text.
"""

FEATURE_GEN_FEEDBACK_PROMPT = """
We trained a RL policy using the reward function generated from the provided reward feature decomposition and tracked the values of the individual components in the reward function as well as global policy metrics such as success rates and episode lengths after every {feedback_subsampling} epochs and the maximum, mean, minimum values encountered:
"""

FEATURE_GEN_INITIAL_PROMPT = """
You are a reward-design assistant for reinforcement learning.

Goal: Decompose the following RL task into a small set of interpretable "features" that capture what humans would consider good performance. These features will later be turned into reward terms and combined as a weighted sum.
""" + FEATURE_GEN_FORMATTING_PROMPT

FEATURE_GEN_EXPLORE_FEEDBACK_PROMPT = """
Analyze each reward component one by one based on the policy feedback and provide a new, improved reward component decomposition that can better solve the task. Some helpful tips for analyzing each the reward components based on the policy feedback:
Some helpful tips for analyzing the policy feedback:
    (1) If the success rates are always near zero, then you must rewrite the whole reward feature decomposition component.
    (2) If the values for a certain reward component are near identical throughout (min ≈ max, range < 0.05). You MUST: 
        (a) Identify the root cause. Is it because the signal is near-constant? Is it because the signal is not correlated with the task success?
        (b) Rethink the signal that is being used to implement the reward component. Change the signal or the way it is being used (e.g., projected_gravity_b[:, 2] for tilt/uprightness).
    (3) If some reward components' magnitude is significantly larger, then you must re-scale to a proper weight.
    (4) If the total reward magnitude grew more than 5x during training (e.g., from ~1 to ~40), the components
        are not properly bounded. Apply torch.tanh or torch.clamp to keep each component in [-1, 1] or [0, 1].
    (5) If the task_score improves initially but then degrades (reward-objective misalignment), strengthen
        the primary_success component weight and/or remove auxiliary terms that pull in a conflicting direction.
    (6) Do NOT reference attributes that are not explicitly provided in the environment's observation method.
Analyze each existing reward component in the suggested manner above first, and then write a brand new reward feature decomposition that has at least two brand new feature.
""" + FEATURE_GEN_FORMATTING_PROMPT

FEATURE_GEN_EXPLOIT_FEEDBACK_PROMPT = """
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
""" + FEATURE_GEN_FORMATTING_PROMPT

# FEATURE_GEN_PROMPT = """
# Decompose the following RL task into a small set of interpretable "features" that capture what humans would consider good performance. These features will later be turned into reward terms and combined as a weighted sum.
# Task context:
# - Task description is: {task_description}
# - The desired task score is: {success_metric_to_win}
# - Here is how we get the observations from the environment: {get_observations_method_as_string}
# """

FEATURE_GEN_PROMPT = """
Decompose the following RL task into a small set of interpretable "features" that capture what humans would consider good performance. These features will later be turned into reward terms and combined as a weighted sum.
Task context:
- Task description is: {task_description}
- Here is how we get the observations from the environment: {get_observations_method_as_string}
"""

############### Single reward function generation templates ###############

FEATURE_AS_ONE_REWARD_FORMATTING_PROMPT = """
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
9) The final reward must remain numerically stable. Avoid unbounded positive terms such as raw norms, raw
   distances, cumulative velocities, or inverse distances without an epsilon and clamp.
10) The keys of individual_rewards_dict must exactly match the feature_name values from FEATURES_JSON. Do not
   add extra diagnostic components to the dictionary unless they correspond to a feature.
11) Before finalizing the code, mentally check two states: an obviously bad state and an obviously good state.
   The good state must receive a higher total reward.

Output requirements:
- Output ONLY a single Python code block.
- The code must define exactly one function:
"""

FEATURE_AS_ONE_REWARD_INITIAL_PROMPT = """
You are a reward engineer for reinforcement learning.
Goal: Write reward functions for an IsaacLab task by turning a given set of human-interpretable features into reward components, then composing them into a final reward as a weighted sum.
""" + FEATURE_AS_ONE_REWARD_FORMATTING_PROMPT

# FEATURE_AS_ONE_REWARD_PROMPT = """
# Write a reward function for the following task: {task_description}
# The desired task score is: {success_metric_to_win}
# Here is how we get the observations from the environment: {get_observations_method_as_string}

# Features to implement (generated previously):
# {FEATURES_JSON}
# """

FEATURE_AS_ONE_REWARD_PROMPT = """
Write a reward function for the following task: {task_description}
Here is how we get the observations from the environment: {get_observations_method_as_string}

Features to implement (generated previously):
{FEATURES_JSON}

Important alignment check:
- Treat the provided desired task score as the target, not necessarily as a value to maximize.
- If the task score is an error/distance with target 0.0, the reward should grow as that error decreases.
- Use each feature's desired_direction on its raw proxy metric, then transform it into a bounded reward
  component with the correct monotonic direction.
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

FEATURE_AS_ONE_FAILURE_FEEDBACK_PROMPT = """
Executing the reward function code above has the following error: {traceback_msg}.
Please fix the bug and provide a new, improved reward function!
""" + FEATURE_AS_ONE_REWARD_FORMATTING_PROMPT

FEATURE_AS_ONE_SUCCESS_PRE_FEEDBACK_PROMPT = """
We trained a RL policy using the provided reward function code and tracked the values of the individual components in the reward function as well as global policy metrics such as success rates and episode lengths after every {feedback_subsampling} epochs and the maximum, mean, minimum values encountered:
"""

FEATURE_AS_ONE_SUCCESS_POST_FEEDBACK_PROMPT = """
Analyze each reward component one by one based on the policy feedback and provide a new, improved reward function that can better solve the task. Some helpful tips for analyzing each of the reward components based on the policy feedback:
    (1) If the values for a certain reward component are near identical throughout (Min ≈ Max, range < 0.05),
        this component is measuring a near-constant signal and provides NO gradient to the policy. You MUST:
        (a) Identify the root cause. Is it because the signal is near-constant? Is it because the signal is not correlated with the task success?
        (b) Rethink the signal that is being used to implement the reward component. Change the signal or the way it is being used (e.g., projected_gravity_b[:, 2] for tilt/uprightness).
    (3) If some reward components' magnitude is significantly larger, then you must re-scale to a proper range.
    (4) If the total reward magnitude grew more than 5x during training (e.g., from ~1 to ~40), the components
        are not properly bounded. Apply torch.tanh or torch.clamp to keep each component in [-1, 1] or [0, 1].
    (5) If the task_score improves initially but then degrades (reward-objective misalignment), strengthen
        the primary_success component weight and/or remove auxiliary terms that pull in a conflicting direction.
    (6) If eureka_total_rewards and task progress are moving in opposite directions, identify the exact component
        causing this and reverse its transformation or reduce its weight. Do not keep an anti-aligned component
        just because it has high magnitude.
    (7) For error-target tasks, compare task_score against the desired value. Reward closeness to the target,
        not the raw task_score value.
Please analyze each existing reward component in the suggested manner above first, and then write the reward function code based on the same set of features.
Features to implement (generated previously): {FEATURES_JSON}
""" + DIRECT_WORKFLOW_REWARD_FORMATTING_INSTRUCTIONS

FEATURE_AS_ONE_NEW_FEATURES_FEEDBACK_PROMPT = """
The previous reward function trained successfully, but it was designed for a different feature set.
Treat it only as implementation reference for:
- valid environment attributes and helper functions;
- tensor shapes and device-safe coding patterns;
- normalization style and reasonable temperature/scale choices.

The refined feature set below is the new source of truth. The new reward function MUST implement this
refined feature set exactly:
- Implement one reward component for every feature_name below, and include every component in
  individual_rewards_dict using the exact feature_name.
- Do NOT include old reward components, old feature names, or old objective terms unless they directly
  match one of the refined features below.
- For each refined feature, infer a measurable scalar signal from the observation method and the feature's
  intent, desired_direction, and typical_failure_mode.
- Convert each signal into a bounded reward component with the correct direction: minimize error/velocity/
  instability signals; maximize only signals that directly represent successful task progress.
- Assign fresh named weights for the refined components based on the provided weights in the FEATURES_JSON.
- If a useful code pattern from the previous reward conflicts with the refined features, discard that
  pattern and follow the refined features.
""" + FEATURE_AS_ONE_REWARD_FORMATTING_PROMPT

HUMAN_RANKING_FEATURE_REFINEMENT_PROMPT = """
Analyze each reward component one by one based on the policy feedback and provide a new, improved reward component decomposition that can better solve the task. Some helpful tips for analyzing each the reward components based on the policy feedback:
Please carefully analyze the policy feedback and provide a new reward feature decomposition that sets the training results as close as possible to the desired task score.
A human observer watched the trained robot policy and ranked the reward features from MOST important to LEAST important based on how much each feature contributes to meaningful, desirable behavior.

Human feature ranking (Rank 1 = most important):
{ranked_feature_list}

Features explicitly dropped by the human (judged unhelpful or harmful):
{dropped_feature_list}

Training performance feedback:
{eureka_task_feedback}

Human textual feedback (highest priority — act on these suggestions):
{human_feedback_section}
Your task is to produce a refined feature set by applying the following rules derived from the human ranking:

RULE 1 — REMOVE all dropped features.
  The human explicitly removed these from the ranking. Do not include them in the refined feature set.
  Exception: if removing them would leave fewer than 2 features, keep the least-dropped one and note it.

RULE 2 — AMPLIFY the highest-ranked feature.
  Multiply its weight by a factor between 1.5× and 3.0×, choosing based on how poorly the
  task is being solved (use a larger multiplier when task_score is far from the desired value).
  Do NOT increase the weight beyond a point where it would numerically dominate all others combined.

RULE 3 — PRESERVE the mid-ranked features.
  Keep their feature_name, intent, measurable_signals, proxy_metric, and desired_direction unchanged.
  You MAY slightly adjust their weights to keep the total weight sum in the original ballpark (within 2×).

RULE 4 — ACT on human feedback.
  If the human provided textual feedback above, treat it as the highest-priority signal.
  Add, modify, or replace features as the human suggests, as long as they are observable from the
  environment's observation method and produce a meaningful proxy metric.

RULE 5 — VALIDATE signal quality.
  Some helpful tips for analyzing the policy feedback and validate the signal quality:
    (1) If the success rates are always near zero, then you must rewrite the whole reward feature decomposition component.
    (2) If the values for a certain reward component are near identical throughout (min ≈ max, range < 0.05). You MUST: 
        (a) Identify the root cause. Is it because the signal is near-constant? Is it because the signal is not correlated with the task success?
        (b) Rethink the signal that is being used to implement the reward component. Change the signal or the way it is being used (e.g., projected_gravity_b[:, 2] for tilt/uprightness).
    (3) If some reward components' magnitude is significantly larger, then you must re-scale to a proper weight.
    (4) If the total reward magnitude grew more than 5x during training (e.g., from ~1 to ~40), the components
        are not properly bounded. Apply torch.tanh or torch.clamp to keep each component in [-1, 1] or [0, 1].
    (5) If the task_score improves initially but then degrades (reward-objective misalignment), strengthen
        the primary_success component weight and/or remove auxiliary terms that pull in a conflicting direction.
    (6) Do NOT reference attributes that are not explicitly provided in the environment's observation method.
    """ + FEATURE_GEN_FORMATTING_PROMPT

DECOMPOSE_REWARD_PROMPT = """
You are a reward engineer for reinforcement learning.
Goal: Implement ONE reward component for ONE specific feature of an IsaacLab RL task. This reward will later be combined with other components in a weighted sum, so it must be well-scaled and interpretable.
Generate Based on the following features: 
<features/>
   feature_name: {feature_name}
   intent: {intent}
   measurable_signals: {measurable_signals}
   proxy_metric: {proxy_metric}
   desired_direction: {desired_direction}
   typical_failure_mode: {typical_failure_mode}
</features>
The task description is: {task_description}
The desired task score is: {success_metric_to_win}
Here is how we get the observations from the environment:
{get_observations_method_as_string}
"""