# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

"""Template strings used for prompting in Isaac Lab Eureka."""


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


DIRECT_WORKFLOW_INITIAL_PROMPT = """
You are a reward engineer trying to write reward functions to solve reinforcement learning tasks as effective as possible.
Your goal is to write a reward function for the environment that will help the agent learn the task described in text.
""" + DIRECT_WORKFLOW_REWARD_FORMATTING_INSTRUCTIONS


TASK_FAILURE_FEEDBACK_PROMPT = """
Executing the reward function code above has the following error: {traceback_msg}.
Please fix the bug and provide a new, improved reward function!
""" + DIRECT_WORKFLOW_REWARD_FORMATTING_INSTRUCTIONS


TASK_SUCCESS_PRE_FEEDBACK_PROMPT = """
We trained a RL policy using the provided reward function code and tracked the values of the individual components in the reward function as well as global policy metrics such as success rates and episode lengths after every {feedback_subsampling} epochs and the maximum, mean, minimum values encountered:
"""


TASK_SUCCESS_POST_FEEDBACK_PROMPT = """
Please carefully analyze the policy feedback and provide a new, improved reward function that can better solve the task. Some helpful tips for analyzing the policy feedback:
    (1) If the success rates are always near zero, then you must rewrite the entire reward function
    (2) If the values for a certain reward component are near identical throughout, then this means RL is not able to optimize this component as it is written. You may consider
        (a) Changing its scale or the value of its temperature parameter
        (b) Re-writing the reward component
        (c) Discarding the reward component
    (3) If some reward components' magnitude is significantly larger, then you must re-scale its value to a proper range
Please analyze each existing reward component in the suggested manner above first, and then write the reward function code.
""" + DIRECT_WORKFLOW_REWARD_FORMATTING_INSTRUCTIONS


DIRECT_WORKFLOW_TASK_PROMPT = """
Write a reward function for the following task: {task_description}
The desired task score is: {success_metric_to_win}
Here is how we get the observations from the environment:
{get_observations_method_as_string}
"""


# FEATURE_GEN_PROMPT = """
# You are a feature engineer trying to generate decompose the reward function into a list of reward components for a reinforcement learning task.
# Your goal is to generate a list of reward components that will help the agent learn the task described in text.
# The task description is: {task_description}
# The desired task score is: {success_metric_to_win}
# Here is how we get the observations from the environment:
# {get_observations_method_as_string}
# """


FEATURE_GEN_FORMATTING_PROMPT = """
Instructions:
1) Propose 6 to 10 candidate features. Each feature must be:
   - Interpretable to a non-expert human (1 sentence description).
   - Measurable from the given observations/actions (no hidden variables).
   - Focused on behavior/outcomes (not “learn faster” or “high return”).
   - As independent as possible (avoid duplicates like “stability” and “uprightness” unless you clearly distinguish them).

2) For each feature, output the following fields:
   - feature_name: short identifier
   - intent: what this feature encourages (1 sentence)
   - measurable_signals: which observation/action variables to use (explicit names from OBS_LIST / ACT_LIST)
   - proxy_metric: a concrete scalar metric formula in plain text (e.g., “abs(pole_angle)”, “-||cart_pos||”, “exp(-k*abs(angle))”)
   - desired_direction: maximize or minimize
   - typical_failure_mode: how an agent could “game” this feature if it were rewarded alone

3) Categorize each feature as one of:
   - primary_success (directly tied to task success)
   - stability_and_safety (constraints, smoothness, avoiding dangerous states)
   - efficiency (energy, action magnitude, time)
   - robustness (recovery after disturbance, maintaining performance under noise)

4) After listing all features, provide:
   - a recommended “starter subset” of 4 to 6 features that likely works well together
   - 2 to 3 alternative subsets emphasizing different human preferences (e.g., “smooth control”, “aggressive recovery”, “energy saving”)

Output format: valid JSON only (no extra commentary).
"""  

FEATURE_GEN_FEEDBACK_PROMPT = """
We trained a RL policy using the reward function generated from the provided reward feature decomposition and tracked the values of the individual components in the reward function as well as global policy metrics such as success rates and episode lengths after every {feedback_subsampling} epochs and the maximum, mean, minimum values encountered:
""" + FEATURE_GEN_FORMATTING_PROMPT

FEATURE_GEN_INITIAL_PROMPT = """
You are a reward-design assistant for reinforcement learning.

Goal: Decompose the following RL task into a small set of interpretable “features” that capture what humans would consider good performance. These features will later be turned into reward terms and combined as a weighted sum.
""" + FEATURE_GEN_FORMATTING_PROMPT

FEATURE_GEN_POST_FEEDBACK_PROMPT = """
Please carefully analyze the policy feedback and provide a new, improved reward component devision that can better solve the task. Some helpful tips for analyzing the policy feedback:
    (1) If the success rates are always near zero, then you must rewrite the entire reward function
    (2) If the values for a certain reward component are near identical throughout, then this means RL is not able to optimize this component as it is written. You may consider
        (a) Changing its scale or the value of its temperature parameter
        (b) Re-writing the reward component
        (c) Discarding the reward component
    (3) If some reward components' magnitude is significantly larger, then you must re-scale its value to a proper range
Please analyze each existing reward component in the suggested manner above first, and then write the reward function code.
""" + FEATURE_GEN_FORMATTING_PROMPT

FEATURE_GEN_PROMPT = """
Decompose the following RL task into a small set of interpretable “features” that capture what humans would consider good performance. These features will later be turned into reward terms and combined as a weighted sum.
Task context:
- Task description is: {task_description}
- The desired task score is: {success_metric_to_win}
- Here is how we get the observations from the environment: {get_observations_method_as_string}
"""

FEATURE_AS_ONE_REWARD_PROMPT = """
You are a reward engineer for reinforcement learning.

Goal: Write reward functions for an IsaacLab task by turning a given set of human-interpretable features into reward components, then composing them into a final reward as a weighted sum.

Task context:
- Task description is: {task_description}
- The desired task score is: {success_metric_to_win}
- Here is how we get the observations from the environment: {get_observations_method_as_string}

Features to implement (generated previously):
{FEATURES_JSON}
Each feature contains:
- feature_name
- intent
- measurable_signals (names from SIGNALS_JSON)
- proxy_metric (plain text formula)
- desired_direction (maximize/minimize)
- typical_failure_mode

Hard requirements:
1) Implement one reward component per feature: r_(feature_name)(...) -> Tensor of shape (num_envs,).
2) Normalize / bound each component to roughly [-1, 1] or [0, 1] when possible (use exp, tanh, or smooth saturation).
3) Avoid sparse-only rewards: each component should provide a learning signal across most states.
4) Mitigate reward hacking: for each feature, add a small safeguard term if needed to prevent its “typical_failure_mode”.
5) Use consistent scaling so no single component dominates by accident.
6) Provide a final reward:
   R = sum_i w_i * r_i
   Use default weights w_i = 1.0 unless you have a strong reason; if you change weights, explain why.

Output requirements:
- Output ONLY a single Python code block.
- The code must define exactly one function:
"""

FEATURE_GEN_FAILURE_FEEDBACK_PROMPT = """
Executing the reward function code above has the following error: {traceback_msg}.
Please fix the bug and provide a new, improved reward function!
""" + FEATURE_AS_ONE_REWARD_PROMPT

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