import datetime
import os
from typing import Literal

# we import this here to avoid GLIBCXX_3.4.30 error in Isaac Sim 5.1
from isaaclab.app import AppLauncher
from isaaclab_eureka import EUREKA_ROOT_DIR
from isaaclab_eureka.config import (
    DIRECT_WORKFLOW_INITIAL_PROMPT,
    DIRECT_WORKFLOW_TASK_PROMPT,
    TASK_FAILURE_FEEDBACK_PROMPT,
    TASK_SUCCESS_POST_FEEDBACK_PROMPT,
    TASK_SUCCESS_PRE_FEEDBACK_PROMPT,
    FEATURE_GEN_PROMPT,
    TASKS_CFG,
    FEATURE_AS_ONE_REWARD_PROMPT,
)
from isaaclab_eureka.managers import EurekaTaskManager, LLMManager
# from isaaclab_eureka.utils import load_tensorboard_logs


gpt_model = "gpt-4"
num_suggestions = 1
temperature = 1.0
task = "Isaac-Cartpole-Direct-v0"
device = "cuda"
env_seed = 42
rl_library = "rsl_rl"
num_processes = 1
max_training_iterations = 100

if task in TASKS_CFG:
    task_description = TASKS_CFG[task]["description"]
    success_metric_string = TASKS_CFG[task].get("success_metric")
    success_metric_to_win = TASKS_CFG[task].get("success_metric_to_win")
    success_metric_tolerance = TASKS_CFG[task].get("success_metric_tolerance")

print("[INFO]: Setting up the LLM Manager...")
llm_manager = LLMManager(
    gpt_model=gpt_model,
    num_suggestions=num_suggestions,
    temperature=temperature,
    system_prompt=DIRECT_WORKFLOW_INITIAL_PROMPT,
    )

print("[INFO]: Setting up the Task Manager...")
task_manager = EurekaTaskManager(
        task=task,
        device=device,
        env_seed=env_seed,
        rl_library=rl_library,
        num_processes=num_processes,
        max_training_iterations=max_training_iterations,
        success_metric_string=success_metric_string,
        )

user_prompt = FEATURE_GEN_PROMPT.format(
    task_description=task_description,
    success_metric_to_win=success_metric_to_win,
    get_observations_method_as_string=task_manager.get_observations_method_as_string
)

print("[INFO]: Prompting the LLM...")
llm_outputs = llm_manager.feature_gen(user_prompt=user_prompt)
feature_strings = llm_outputs["feature_strings"]
reward_code = llm_manager.prompt(user_prompt=FEATURE_AS_ONE_REWARD_PROMPT.format(
    task_description=task_description,
    success_metric_to_win=success_metric_to_win,
    get_observations_method_as_string=task_manager.get_observations_method_as_string,
    FEATURES_JSON=feature_strings,
))
print(reward_code["reward_strings"])

task_manager.close()

# print(gpt_reward_method_strings)
