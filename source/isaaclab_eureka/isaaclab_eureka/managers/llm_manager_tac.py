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
                                    # "measurable_signals": {
                                    #     "description": "Signals that can be used to measure the feature.",
                                    #     "type": "array",
                                    #     "items": {
                                    #         "type": "string",
                                    #     },
                                    # },
                                    # "proxy_metric": {
                                    #     "description": "Proxy metric for the feature. Explain the proxy metric for the feature in one sentence.",
                                    #     "type": "string",
                                    # },
                                    # "weight": {
                                    #     "description": "Weight of the feature. The weight of the feature in the final reward function.",
                                    #     "type": "number",
                                    # },
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
