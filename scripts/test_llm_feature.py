# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

import re
import json

from ._llm_request_utils import (
    build_openai_client,
    create_chat_completion,
    get_llm_request_settings,
)


class LLMManagerTac:
    """Manager to interface with the LLM API.

    This class is responsible for interfacing with the LLM API to generate rewards.
    It establishes a connection either to native OpenAI API, or to the Azure OpenAI API.

    The Openai API relies on the following environment variables to be set:
    - For the native OpenAI API, the environment variable OPENAI_API_KEY must be set.
    - For the Azure OpenAI API, the environment variables AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set.
    """

    def __init__(self, gpt_model: str, temperature: float, system_prompt: str, feature_prompt: str = None):
        """Initialize the LLMManager

        Args:
            gpt_model: The model to use for the LLM API
            num_suggestions: The number of independent suggestions to generate
            temperature: The temperature to use for the LLM API
            system_prompt: The system prompt to provide to the LLM API
        """

        self._gpt_model = gpt_model
        self._num_suggestions = 1
        self._temperature = temperature
        self._reward_system_prompt = system_prompt
        self._feature_system_prompt = feature_prompt
        self._prompts = [{"role": "system", "content": system_prompt}]
        self._feature_prompts = [{"role": "system", "content": feature_prompt}]
        self._single_feature_reward_generation_prompts = [{"role": "system", "content": system_prompt}]
        (
            self._request_timeout_seconds,
            self._max_request_retries,
            self._retry_backoff_seconds,
        ) = get_llm_request_settings()
        self._client = build_openai_client(timeout_seconds=self._request_timeout_seconds)

    def extract_code_from_response(self, response: str) -> str:
        """Extract the code component from the LLM response

        If the response contains a code block of the form "```python ... ```", extract the code block from the response.
        Otherwise, return an empty string.

        Args:
            response: The response from the LLM API
        """
        pattern = r"```python(.*?)```"
        result = re.findall(pattern, response, re.DOTALL)
        code_string = ""
        if result is not None and len(result) > 0:
            code_string = result[-1]
            # Remove leading newline characters
            code_string = code_string.lstrip("\n")
        return code_string
    
    def extract_json_from_response(self, response: str):
        feature_dic = json.loads(response)
        features = feature_dic["features"]
        return features

    def feature_gen(self, user_prompt: str, assistant_prompt: str = None, num_suggestion: int = 1) -> list[str]:
        """Call the LLM API to generate features

        Args:
            user_prompt: The user prompt to provide to the LLM API

        Returns:
            A dictionary containing the feature strings and raw outputs from the LLM
        """
        if assistant_prompt is not None:
            self._feature_prompts.append({"role": "assistant", "content": assistant_prompt})
        self._feature_prompts.append({"role": "user", "content": user_prompt})


        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "feature_list",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "features": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 4,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": [
                                    "feature_name",
                                    "intent",
                                    "measurable_signals",
                                    "proxy_metric",
                                    "weight",
                                    "desired_direction",
                                    "typical_failure_mode",
                                ],
                                "properties": {
                                    "feature_name": {
                                        "description": "Name of the feature that is measurable from the given observations/actions.",
                                        "type": "string",
                                    },
                                    "intent": {
                                        "description": "Intent of the feature. Explain the intent of the feature in one sentence.",
                                        "type": "string",
                                    },
                                    "measurable_signals": {
                                        "description": "Signals that can be used to measure the feature.",
                                        "type": "array",
                                        "items": {
                                            "type": "string",
                                        },
                                    },
                                    "proxy_metric": {
                                        "description": "Proxy metric for the feature. Explain the proxy metric for the feature in one sentence.",
                                        "type": "string",
                                    },
                                    "weight": {
                                        "description": "Weight of the feature. The weight of the feature in the final reward function.",
                                        "type": "number",
                                    },
                                    "desired_direction": {
                                        "description": "Desired direction of the feature. Explain the desired direction of the feature in one sentence.",
                                        "type": "string",
                                    },
                                    "typical_failure_mode": {
                                        "description": "Typical failure mode of the feature. Explain the typical failure mode of the feature in one sentence.",
                                        "type": "string",
                                    },
                                },
                            },
                        }
                    },
                    "required": ["features"],
                    "additionalProperties": False,
                }
            }
        }

        # The official Eureka code only keeps the last round of feedback
        if len(self._feature_prompts) == 6:
            self._feature_prompts.pop(2)
            self._feature_prompts.pop(2)
        try:
            responses = create_chat_completion(
                self._client,
                model=self._gpt_model,
                messages=self._feature_prompts,
                temperature=self._temperature,
                n=num_suggestion,
                response_format=response_format,
                timeout_seconds=self._request_timeout_seconds,
                max_request_retries=self._max_request_retries,
                retry_backoff_seconds=self._retry_backoff_seconds,
                request_name="LLMManagerTac.feature_gen",
            )
        except Exception as e:
            raise RuntimeError("An error occurred while prompting the LLM") from e
        
        raw_outputs = [response.message.content for response in responses.choices]
        try:
            feature_strings = [self.extract_json_from_response(raw_output) for raw_output in raw_outputs]
        except Exception as e:
            # raise RuntimeError("An error occurred while extracting the feature strings") from e
            print(f"An error occurred while extracting the feature strings: {e}", e)
            feature_strings = raw_outputs
        # print("--------------------------------FEATURE STRINGS--------------------------------")
        # print(feature_strings)
        return {"feature_strings": feature_strings, "raw_outputs": raw_outputs}

    def prompt(self, user_prompt: str, assistant_prompt: str = None) -> list[str]:
        """Call the LLM API to collect responses

        Args:
            user_prompt: The user prompt to provide to the LLM API
            assistant_prompt: The assistant prompt to provide to the LLM API

        Returns:
            A dictionary containing the reward strings and raw outputs from the LLM

        Raises:
            Exception: If there is an error with the LLM API
        """
        if assistant_prompt is not None:
            self._prompts.append({"role": "assistant", "content": assistant_prompt})
        self._prompts.append({"role": "user", "content": user_prompt})

        # The official Eureka code only keeps the last round of feedback
        if len(self._prompts) == 6:
            self._prompts.pop(2)
            self._prompts.pop(2)

        try:
            responses = create_chat_completion(
                self._client,
                model=self._gpt_model,
                messages=self._prompts,
                temperature=self._temperature,
                n=self._num_suggestions,
                timeout_seconds=self._request_timeout_seconds,
                max_request_retries=self._max_request_retries,
                retry_backoff_seconds=self._retry_backoff_seconds,
                request_name="LLMManagerTac.prompt",
            )
        except Exception as e:
            raise RuntimeError("An error occurred while prompting the LLM") from e

        raw_outputs = [response.message.content for response in responses.choices]
        reward_strings = [self.extract_code_from_response(raw_output) for raw_output in raw_outputs]
        return {"reward_strings": reward_strings, "raw_outputs": raw_outputs}

    def single_feature_prompt(self, user_prompt: str, assistant_prompt: str = None, num_suggestion: int = 1) -> list[str]:
        """Call the LLM API to collect responses

        Args:
            user_prompt: The user prompt to provide to the LLM API
            assistant_prompt: The assistant prompt to provide to the LLM API

        Returns:
            A dictionary containing the reward strings and raw outputs from the LLM

        Raises:
            Exception: If there is an error with the LLM API
        """
        # self._single_feature_reward_generation_prompts = self._prompts.copy()
        if assistant_prompt is not None:
            self._single_feature_reward_generation_prompts .append({"role": "assistant", "content": assistant_prompt})
        self._single_feature_reward_generation_prompts.append({"role": "user", "content": user_prompt})

        # The official Eureka code only keeps the last round of feedback
        if len(self._single_feature_reward_generation_prompts) == 6:
            self._single_feature_reward_generation_prompts.pop(2)
            # self._single_feature_reward_generation_prompts.pop(2)

        try:
            responses = create_chat_completion(
                self._client,
                model=self._gpt_model,
                messages=self._single_feature_reward_generation_prompts,
                temperature=self._temperature,
                n=num_suggestion,
                timeout_seconds=self._request_timeout_seconds,
                max_request_retries=self._max_request_retries,
                retry_backoff_seconds=self._retry_backoff_seconds,
                request_name="LLMManagerTac.single_feature_prompt",
            )
        except Exception as e:
            raise RuntimeError("An error occurred while prompting the LLM") from e

        raw_outputs = [response.message.content for response in responses.choices]
        reward_strings = [self.extract_code_from_response(raw_output) for raw_output in raw_outputs]
        # print("--------------------------------REWARD GENERATION PROMPTS--------------------------------")
        # print(self._single_feature_reward_generation_prompts)
        # print("--------------------------------REWARD STRINGS GENERATED--------------------------------")
        # print(reward_strings)
        return {"reward_strings": reward_strings, "raw_outputs": raw_outputs}
    
    def single_feature_reset(self):
        """Reset the single-feature reward generation conversation back to just the system prompt."""
        self._single_feature_reward_generation_prompts = [
            {"role": "system", "content": self._reward_system_prompt}
        ]

    def refine_single_feature(
        self,
        user_prompt: str,
        system_prompt: str,
        num_suggestion: int = 1,
    ) -> dict:
        """Refine ONE feature in a locked feature set via a stateless LLM call.

        Each call is independent (no conversation history is retained) because the
        per-component refinement context is fully provided in `user_prompt`. The
        response is forced to a strict JSON schema describing a single refined feature.

        Args:
            user_prompt: The user prompt describing the feature being refined and its
                observed component-level + task-level feedback.
            system_prompt: System prompt describing the refinement role/constraints.
            num_suggestion: Number of independent refinements to sample from the LLM.

        Returns:
            A dict with:
              - "refined_features": list of parsed feature dicts (one per suggestion)
              - "raw_outputs": list of raw string responses from the LLM
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "single_refined_feature",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "feature_name",
                        "intent",
                        "desired_direction",
                        "typical_failure_mode",
                    ],
                    "properties": {
                        "feature_name": {
                            "description": "MUST match the locked feature_name being refined.",
                            "type": "string",
                        },
                        "intent": {
                            "description": "Refined one-sentence description of what this feature encourages.",
                            "type": "string",
                        },
                        "desired_direction": {
                            "description": "Refined desired direction (maximize or minimize).",
                            "type": "string",
                        },
                        "typical_failure_mode": {
                            "description": "Refined typical failure mode for this feature.",
                            "type": "string",
                        },
                    },
                },
            },
        }

        try:
            responses = create_chat_completion(
                self._client,
                model=self._gpt_model,
                messages=messages,
                temperature=self._temperature,
                n=num_suggestion,
                response_format=response_format,
                timeout_seconds=self._request_timeout_seconds,
                max_request_retries=self._max_request_retries,
                retry_backoff_seconds=self._retry_backoff_seconds,
                request_name="LLMManagerTac.refine_single_feature",
            )
        except Exception as e:
            raise RuntimeError("An error occurred while prompting the LLM for feature refinement") from e

        raw_outputs = [response.message.content for response in responses.choices]
        refined_features = []
        for raw_output in raw_outputs:
            try:
                refined_features.append(json.loads(raw_output))
            except Exception as e:
                print(f"[WARNING] Failed to parse refined-feature JSON: {e}. Raw output: {raw_output}")
                refined_features.append(None)

        return {"refined_features": refined_features, "raw_outputs": raw_outputs}


if __name__ == "__main__":
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
    FEATURE_GEN_PROMPT = """
    Decompose the following RL task into a small set of interpretable "features" that capture what humans would consider good performance. These features will later be turned into reward terms and combined as a weighted sum.
        Task context:
            - Task description is: {task_description}
            - Here is how we get the observations from the environment: {get_observations_method_as_string}
            """
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
    feature_gen_prompt = FEATURE_GEN_PROMPT.format(
            task_description=task_description,
            # success_metric_to_win=self._success_metric_to_win,
            get_observations_method_as_string=task_manager.get_observations_method_as_string,
        )
    self._llm_manager = LLMManagerTac(
            gpt_model="gpt-4o-mini",
            temperature=1,
            system_prompt=FEATURE_AS_ONE_REWARD_INITIAL_PROMPT,
            feature_prompt=None,
        )
    feature_gen_outputs = llm_manager.feature_gen(user_prompt=feature_gen_prompt, assistant_prompt=None, num_suggestion= 1)
    print(feature_gen_outputs)
