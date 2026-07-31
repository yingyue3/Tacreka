import datetime
import json
import os
from typing import Literal

# we import this here to avoid GLIBCXX_3.4.30 error in Isaac Sim 5.1
from isaaclab.app import AppLauncher
from isaaclab_eureka import EUREKA_ROOT_DIR
from isaaclab_eureka.config import (
    DIRECT_WORKFLOW_INITIAL_PROMPT,
    DIRECT_WORKFLOW_TASK_PROMPT,
    TASKS_CFG,
    FEATURE_GEN_FORMATTING_PROMPT,
    FEATURE_GEN_FEEDBACK_PROMPT,
    FEATURE_GEN_INITIAL_PROMPT,
    FEATURE_GEN_PROMPT,
    FEATURE_AS_ONE_REWARD_PROMPT,
    FEATURE_AS_ONE_REWARD_INITIAL_PROMPT,
    FEATURE_GEN_EXPLORE_FEEDBACK_PROMPT,
    FEATURE_GEN_EXPLOIT_FEEDBACK_PROMPT,
    FEATURE_AS_ONE_FAILURE_FEEDBACK_PROMPT,
    FEATURE_AS_ONE_SUCCESS_POST_FEEDBACK_PROMPT,
    FEATURE_AS_ONE_SUCCESS_PRE_FEEDBACK_PROMPT,
    TASK_SUCCESS_PRE_FEEDBACK_PROMPT,
    FEATURE_AS_ONE_NEW_FEATURES_FEEDBACK_PROMPT
)
from isaaclab_eureka.managers import LLMManagerTac
# from isaaclab_eureka.utils import load_tensorboard_logs


gpt_model = "gpt-4o-mini"
num_suggestions = 1
temperature = 1.0
task = "Merge_Highway"
device = "cuda"
env_seed = 42
rl_library = "rsl_rl"
num_processes = 1
max_training_iterations = 100


print("[INFO]: Setting up the LLM Manager...")
llm_manager = LLMManagerTac(
    gpt_model=gpt_model,
    temperature=temperature,
    system_prompt=FEATURE_AS_ONE_REWARD_INITIAL_PROMPT,
    feature_prompt=FEATURE_GEN_INITIAL_PROMPT,
    )

task_description = "In this task, the ego-vehicle starts on a main highway but soon approaches a road junction with incoming vehicles on the access ramp. The agent’s objective is now to maintain a high speed while making room for the vehicles so that they can safely merge in the traffic."
get_observations_method_as_string = """{
    "observation": {
        "type": "Kinematics"
    },
    "action": {
        "type": "DiscreteMetaAction"
    },
    "collision_reward": -1,
    "right_lane_reward": 0.1,
    "high_speed_reward": 0.2,
    "reward_speed_range": [20, 30],
    "merging_speed_reward": -0.5,
    "lane_change_reward": -0.05,
    "simulation_frequency": 15,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,  # [px]
    "screen_height": 150,  # [px]
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": None
    }"""

user_prompt = FEATURE_GEN_PROMPT.format(
    task_description=task_description,
    get_observations_method_as_string=get_observations_method_as_string)

print("[INFO]: Prompting the LLM...")
llm_outputs = llm_manager.feature_gen(user_prompt=user_prompt, assistant_prompt=None, num_suggestion= 1)
# feature_strings = llm_outputs["feature_strings"][0]
# reward_code = llm_manager.prompt(user_prompt=FEATURE_AS_ONE_REWARD_PROMPT.format(
#     task_description=task_description,
#     get_observations_method_as_string=get_observations_method_as_string,
#     FEATURES_JSON=json.dumps(feature_strings, indent=2),
# ))
print(llm_outputs["feature_strings"][0])

# print(gpt_reward_method_strings)
