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

   CRITICAL signal-selection rules — violating these produces useless or misleading reward components:
   a) Do NOT use the Euclidean norm of a vector whose magnitude is approximately constant.
      Example: torch.norm(projected_gravity_b, dim=-1) is always approximately 9.81 m/s^2 regardless
      of the robot's orientation — it carries ZERO information about tilt or uprightness.
      For uprightness/tilt, use a DIRECTIONAL component instead:
        projected_gravity_b[:, 2]  (body-frame z-projection; near -9.81 when level, near 0 when severely tilted).
   b) Do NOT reference environment attributes that are not explicitly listed in the provided observation
      method (e.g., do not invent "previous_root_lin_vel_b" if it is not in the env class).
   c) Confirm that your chosen signal differs significantly between a "good" agent state and a "bad" agent
      state. If the signal_range min and max are very close (< 0.1 difference), pick a different signal.

3) Categorize each feature as one of:
   - primary_success (directly tied to task success)
   - stability_and_safety (constraints, smoothness, avoiding dangerous states)
   - efficiency (energy, action magnitude, time)
   - robustness (recovery after disturbance, maintaining performance under noise)

4) After listing all features, provide:
   - a recommended "starter subset" of 3 to 6 features that likely works well together
   - 2 to 3 alternative subsets emphasizing different human preferences (e.g., "smooth control", "aggressive recovery", "energy saving")

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
Please carefully analyze the policy feedback and provide a new reward feature decomposition that sets the training results as close as possible to the desired task score. The new reward feature decomposition should have at least one brand new feature.
Some helpful tips for analyzing the policy feedback:
    (1) If the success rates are always near zero, then you must add a new reward feature decomposition component or consider dropping existing ones.
    (2) If the values for a certain reward component are near identical throughout (min ≈ max, range < 0.05),
        this component is measuring a near-constant signal and provides NO gradient to the policy. You MUST:
        (a) Identify why the signal is constant (e.g., mistakenly using the norm of a fixed-magnitude vector).
        (b) Replace it with a directional component or a signal that genuinely varies with agent behavior.
            For example, replace torch.norm(projected_gravity_b) with projected_gravity_b[:, 2] for tilt.
        (c) Verify the new feature has a meaningfully wide signal_range between good and bad agent states.
    (3) Do NOT reference attributes that are not explicitly provided in the environment's observation method.
Please analyze each existing reward component in the suggested manner above first, and then write the reward feature decomposition.
""" + FEATURE_GEN_FORMATTING_PROMPT

FEATURE_GEN_EXPLOIT_FEEDBACK_PROMPT = """
Please carefully analyze the policy feedback and provide a new reward feature decomposition that sets the training results as close as possible to the desired task score. The new reward feature decomposition should have at least one feature with changed weights or discard one of the existing features.
Some helpful tips for analyzing the policy feedback:
    (1) If the success rates are always near zero, then consider dropping unhelpful reward feature components.
    (2) If the values for a certain reward component are near identical throughout (range < 0.05), this component
        is measuring a near-constant signal. You may consider:
        (a) Changing its temperature parameter to produce a wider output range
        (b) Discarding the reward feature entirely and replacing with one that has a non-trivial signal_range
    (3) If some reward feature magnitudes are significantly larger than others, re-scale to a comparable range.
    (4) If the total reward grew more than 5x during training, the features are not properly bounded;
        add normalization constraints (e.g., torch.tanh or torch.clamp) in the implementation.
Please analyze each existing reward feature in the suggested manner above first, and then write the reward feature decomposition.
""" + FEATURE_GEN_FORMATTING_PROMPT

FEATURE_GEN_PROMPT = """
Decompose the following RL task into a small set of interpretable "features" that capture what humans would consider good performance. These features will later be turned into reward terms and combined as a weighted sum.
Task context:
- Task description is: {task_description}
- The desired task score is: {success_metric_to_win}
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

FEATURE_AS_ONE_REWARD_INITIAL_PROMPT = """
You are a reward engineer for reinforcement learning.
Goal: Write reward functions for an IsaacLab task by turning a given set of human-interpretable features into reward components, then composing them into a final reward as a weighted sum.
""" + FEATURE_AS_ONE_REWARD_FORMATTING_PROMPT

FEATURE_AS_ONE_REWARD_PROMPT = """
Write a reward function for the following task: {task_description}
The desired task score is: {success_metric_to_win}
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

FEATURE_AS_ONE_FAILURE_FEEDBACK_PROMPT = """
Executing the reward function code above has the following error: {traceback_msg}.
Please fix the bug and provide a new, improved reward function!
""" + FEATURE_AS_ONE_REWARD_FORMATTING_PROMPT

FEATURE_AS_ONE_SUCCESS_PRE_FEEDBACK_PROMPT = """
We trained a RL policy using the provided reward function code and tracked the values of the individual components in the reward function as well as global policy metrics such as success rates and episode lengths after every {feedback_subsampling} epochs and the maximum, mean, minimum values encountered:
"""

FEATURE_AS_ONE_SUCCESS_POST_FEEDBACK_PROMPT = """
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

HUMAN_RANKING_FEATURE_REFINEMENT_PROMPT = """
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
  For any retained or newly added feature, verify:
  (a) Its measurable_signals exist in the environment observation method.
  (b) Its proxy_metric produces a signal that varies meaningfully between good and bad agent states.
  (c) It does NOT use torch.norm(projected_gravity_b) as an uprightness proxy (use projected_gravity_b[:, 2] instead).
  If a retained feature fails validation, fix the proxy_metric or replace the feature with a corrected one.
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