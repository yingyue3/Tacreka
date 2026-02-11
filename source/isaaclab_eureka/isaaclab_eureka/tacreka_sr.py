# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

import datetime
import os
import json
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
    FEATURE_GEN_POST_FEEDBACK_PROMPT,
    FEATURE_AS_ONE_FAILURE_FEEDBACK_PROMPT,
    FEATURE_AS_ONE_SUCCESS_POST_FEEDBACK_PROMPT,
    TASK_SUCCESS_PRE_FEEDBACK_PROMPT
)
from isaaclab_eureka.managers import EurekaTaskManager, LLMManager
from isaaclab_eureka.utils import load_tensorboard_logs


class Tacreka_SR:
    """Orchestrates the training of the RL agent using the LLM."""

    def __init__(
        self,
        task: str,
        device: str = "cuda",
        env_seed: int = 42,
        rl_library: Literal["rsl_rl", "rl_games"] = "rsl_rl",
        max_training_iterations: int = 100,
        feedback_subsampling: int = 10,
        temperature: float = 1.0,
        gpt_model: str = "gpt-4",
        num_parallel_runs: int = 2,
        use_wandb: bool = True,
        wandb_project: str = "isaaclab-eureka",
        wandb_entity: str = None,
        wandb_name: str = None,
    ):
        """Initialize the Eureka class.

        Args:

            task: The task to train the agent on.
            device: The device to run the training on.
            env_seed: The seed to use for the environment
            rl_library: The RL library to use for training.
            max_training_iterations: The maximum number of training iterations for the RL agent.
            feedback_subsampling: The subsampling of the metrics given as feedack to the LLM.
            temperature: The temperature to use for the GPT model.
            gpt_model: The GPT model to use.
            num_parallel_runs: The number of runs to execute in parallel.
            use_wandb: Whether to use Weights & Biases for logging.
            wandb_project: The wandb project name.
            wandb_entity: The wandb entity/team name.
            wandb_name: The wandb run name. If None, uses timestamp.
        """

        # Load the task description and success metric
        if task in TASKS_CFG:
            task_description = TASKS_CFG[task]["description"]
            success_metric_string = TASKS_CFG[task].get("success_metric")
            self._success_metric_to_win = TASKS_CFG[task].get("success_metric_to_win")
            self._success_metric_tolerance = TASKS_CFG[task].get("success_metric_tolerance")
        else:
            raise ValueError(
                f"Task configuration for {task} not found in the `TASKS_CFG` dictionary in config/tasks.py."
            )

        self._task_description = task_description
        self._feedback_subsampling = feedback_subsampling
        self._num_processes = num_parallel_runs
        self._num_parallel_runs = 2

        print("[INFO]: Setting up the LLM Manager...")
        self._llm_manager = LLMManager(
            gpt_model=gpt_model,
            num_suggestions=self._num_processes,
            temperature=temperature,
            system_prompt=FEATURE_AS_ONE_REWARD_INITIAL_PROMPT,
            feature_prompt=FEATURE_GEN_INITIAL_PROMPT,
        )

        print("[INFO]: Setting up the Task Manager...")
        self._task_manager = EurekaTaskManager(
            task=task,
            device=device,
            env_seed=env_seed,
            rl_library=rl_library,
            num_processes=self._num_parallel_runs,
            max_training_iterations=max_training_iterations,
            success_metric_string=success_metric_string,
        )

        # Logging
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._log_dir = os.path.join(EUREKA_ROOT_DIR, "logs", "eureka", task, timestamp)
        os.makedirs(self._log_dir)

        # We import here because doing this before launching Kit causes GLIBCXX errors
        from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

        self._tensorboard_writer = TensorboardSummaryWriter(log_dir=self._log_dir, flush_secs=10)
        
        # Initialize wandb if requested
        self._use_wandb = use_wandb
        self._wandb = None
        if use_wandb:
            try:
                import wandb
                self._wandb = wandb
                
                # Determine run name
                run_name = wandb_name if wandb_name else f"{task}_{timestamp}"
                
                # Initialize wandb
                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=run_name,
                    config={
                        "task": task,
                        "device": device,
                        "env_seed": env_seed,
                        "rl_library": rl_library,
                        "max_training_iterations": max_training_iterations,
                        "feedback_subsampling": feedback_subsampling,
                        "temperature": temperature,
                        "gpt_model": gpt_model,
                        "num_parallel_runs": num_parallel_runs,
                        "task_description": task_description,
                        "success_metric_to_win": self._success_metric_to_win,
                        "success_metric_tolerance": self._success_metric_tolerance,
                    },
                    dir=self._log_dir,
                )
                print(f"[INFO]: Weights & Biases logging initialized. Project: {wandb_project}, Run: {run_name}")
            except ImportError:
                print("[WARNING]: wandb not installed. Install with 'pip install wandb' to enable wandb logging.")
                self._use_wandb = False
                self._wandb = None
    
    user_feedback_prompt_rw_gen = None
    def run(self, max_eureka_iterations: int):
        """Run the Eureka training loop.

        Args:
            max_eureka_iterations: The maximum number of Eureka iterations to run.
        """
        # We import here because doing this before launching Kit causes GCC_12.0 errors
        import numpy as np

        # Initial prompts
        feature_gen_prompt = FEATURE_GEN_PROMPT.format(
            task_description=self._task_description,
            success_metric_to_win=self._success_metric_to_win,
            get_observations_method_as_string=self._task_manager.get_observations_method_as_string,
        )
        # The assistant prompt is used to feed the previous LLM output back into the LLM
        assistant_prompt = None
        rw_gen_assistant_prompt = None

        # The best run across all iterations
        best_run_results = {"success_metric": None}

        for iter in range(max_eureka_iterations):
            print(f"\n{'#' * 20} Running Eureka Iteration {iter} {'#' * 20} \n")
            # Generate the GPT reward methods
            if assistant_prompt != "N":
                feature_gen_outputs = self._llm_manager.feature_gen(user_prompt=feature_gen_prompt, assistant_prompt=assistant_prompt)
                feature_strings = feature_gen_outputs["feature_strings"]
            # self._llm_manager.single_feature_reset()
            print(f"\n{'+' * 20} Feature Generated {'+' * 20} \n")
            llm_outputs = []
            gpt_reward_method_strings = []
            for idx, feature_string in enumerate(feature_strings):

                reward_code = self._llm_manager.single_feature_prompt(user_prompt=FEATURE_AS_ONE_REWARD_PROMPT.format(
                    task_description=self._task_description,
                    success_metric_to_win=self._success_metric_to_win,
                    get_observations_method_as_string=self._task_manager.get_observations_method_as_string,
                    FEATURES_JSON=feature_string,
                ), assistant_prompt=rw_gen_assistant_prompt, 
                num_suggestion= self._num_parallel_runs, # only one suggestion is needed for the single feature prompt
                )
                llm_outputs.append(reward_code)
            # Log the llm outputs
            i = 0
            for idx, llm_output in enumerate(llm_outputs):
                raw_outputs = llm_output["raw_outputs"]
                reward_strings = llm_output["reward_strings"]
                for idx_raw, raw_output in enumerate(raw_outputs):
                    i += 1
                    self._tensorboard_writer.add_text(f"Run_{i}/raw_llm_output", raw_output, iter)
                    self._tensorboard_writer.add_text(f"Run_{i}/feature_idx", str(idx), iter)
                    gpt_reward_method_strings.append({"reward_strings" : reward_strings[idx_raw], "feature_idx" : idx, "raw_output" : raw_output})

            # Train the RL agent
            results = []
            for llm_output in llm_outputs:
                results += self._task_manager.train(llm_output["reward_strings"])
            # Give TensorBoard time to flush logs before reading them
            import time
            time.sleep(1.0)  # Wait 1 second for TensorBoard to flush
            # Evaluate the results
            iter_best_success_metric = None
            best_run_idx = 0
            for idx, result in enumerate(results):
                if not result["success"]:
                    user_feedback_prompt_rw_gen = FEATURE_AS_ONE_FAILURE_FEEDBACK_PROMPT.format(traceback_msg=result["exception"])
                    user_feedback_prompt = "N"
                    print("Failed to generate correct reward function, using previous feedback prompt")
                else:
                    # Compute the performance metrics
                    print("Successfully generated reward function, generating task feedback")
                    eureka_task_feedback, success_metric_max, rewards_correlation = self._get_eureka_task_feedback(
                        result["log_dir"], self._feedback_subsampling
                    )

                    # Generate the user feedback prompt
                    user_feedback_prompt = (
                        FEATURE_GEN_FEEDBACK_PROMPT.format(feedback_subsampling=self._feedback_subsampling)
                        + eureka_task_feedback
                        + FEATURE_GEN_POST_FEEDBACK_PROMPT
                    )

                    # user_feedback_prompt_rw_gen = (
                    #     TASK_SUCCESS_PRE_FEEDBACK_PROMPT.format(feedback_subsampling=self._feedback_subsampling)
                    #     + eureka_task_feedback
                    #     + FEATURE_AS_ONE_SUCCESS_POST_FEEDBACK_PROMPT
                    # )
                    user_feedback_prompt_rw_gen = None

                    # Store the results
                    results[idx]["eureka_task_feedback"] = eureka_task_feedback
                    results[idx]["success_metric_max"] = success_metric_max
                    results[idx]["rewards_correlation"] = rewards_correlation
                    feature_idx = gpt_reward_method_strings[idx]["feature_idx"]
                    results[idx]["reward_components"] = feature_gen_outputs["raw_outputs"][feature_idx]
                    # Log metrics to wandb
                    if self._use_wandb and self._wandb:
                        self._wandb.log({
                            f"Run_{idx}/best_success_metric": success_metric_max if success_metric_max is not None else 0.0,
                            f"Run_{idx}/rewards_correlation": rewards_correlation,
                        }, step=iter)
                    # Check the best performing metric, determined by the minimum distance from the win target
                    if success_metric_max is not None and (
                        iter_best_success_metric is None
                        or np.abs(success_metric_max - self._success_metric_to_win)
                        < np.abs(iter_best_success_metric - self._success_metric_to_win)
                    ):
                        # Store the best run for this iteration
                        iter_best_success_metric = success_metric_max
                        best_run_idx = idx

                        # Store the best metric across all iterations
                        if best_run_results["success_metric"] is None or (
                            np.abs(iter_best_success_metric - self._success_metric_to_win)
                            < np.abs(best_run_results["success_metric"] - self._success_metric_to_win)
                        ):
                            best_run_results["success_metric"] = iter_best_success_metric
                            best_run_results["gpt_reward_method"] = gpt_reward_method_strings[idx]["reward_strings"]
                            best_run_results["feature_idx"] = gpt_reward_method_strings[idx]["feature_idx"]
                            best_run_results["task_feedback"] = eureka_task_feedback
                            print("logging best metric to wandb")
                            # Log best metric to wandb
                            if self._use_wandb and self._wandb:
                                self._wandb.log({
                                    "best/overall_success_metric": iter_best_success_metric,
                                    "best/iteration": iter,
                                    "best/run_idx": idx,
                                }, step=iter)

                # Add the prompts
                feature_idx = gpt_reward_method_strings[idx]["feature_idx"]
                results[idx]["user_prompt"] = user_feedback_prompt
                results[idx]["assistant_prompt_rw_gen"] = user_feedback_prompt_rw_gen
                results[idx]["assistant_prompt"] = feature_gen_outputs["raw_outputs"][feature_idx]
                results[idx]["feature_idx"] = feature_idx

            self._log_iteration_results(iter, results)

            if (
                best_run_results["success_metric"] is not None
                and np.abs(best_run_results["success_metric"] - self._success_metric_to_win)
                < self._success_metric_tolerance
            ):
                print(f"Task solved with success metric: {best_run_results['success_metric']}")
                break

            assistant_prompt = results[best_run_idx]["assistant_prompt"]
            feature_gen_prompt = results[best_run_idx]["user_prompt"]
            rw_gen_assistant_prompt = results[best_run_idx]["assistant_prompt_rw_gen"]

        self._log_final_results(best_run_results)
        # Close the task manager
        self._task_manager.close()

    def _get_eureka_task_feedback(self, log_dir: str, feedback_subsampling: int) -> tuple[str, float, float]:
        """Get the feedback for the Eureka task.

        Args:
            log_dir: The directory where the tensorboard logs are stored.
            feedback_subsampling: The subsampling of the metrics' trajectories.
        Returns:
            A tuple containing the feedback string, the maximum of the success metric, and the correlation between the oracle and GPT rewards.
        """
        # We import here because doing this before launching Kit causes GCC_12.0 errors
        import numpy as np

        data = load_tensorboard_logs(log_dir)

        # Compute correlation between the oracle and GPT rewards
        eureka_rewards_data = next((data[key] for key in data if key.endswith("Eureka/eureka_total_rewards")), None)
        oracle_rewards_data = next((data[key] for key in data if key.endswith("Eureka/oracle_total_rewards")), None)
        
        # Handle case where rewards data is missing
        if eureka_rewards_data is None or oracle_rewards_data is None:
            print(f"[WARNING] Missing reward data in TensorBoard logs. Available keys: {list(data.keys())}")
            print(f"[WARNING] Eureka rewards found: {eureka_rewards_data is not None}, Oracle rewards found: {oracle_rewards_data is not None}")
            # Return default correlation of 0.0 if data is missing
            rewards_correlation = 0.0
        else:
            eureka_rewards = np.array(eureka_rewards_data)
            oracle_rewards = np.array(oracle_rewards_data)
            
            # Check if arrays have valid shape
            if eureka_rewards.ndim == 0 or oracle_rewards.ndim == 0:
                print(f"[WARNING] Reward arrays have invalid shape. Eureka: {eureka_rewards.shape}, Oracle: {oracle_rewards.shape}")
                rewards_correlation = 0.0
            elif len(eureka_rewards) == 0 or len(oracle_rewards) == 0:
                print(f"[WARNING] Reward arrays are empty. Eureka: {len(eureka_rewards)}, Oracle: {len(oracle_rewards)}")
                rewards_correlation = 0.0
            else:
                # Sometimes, the tensorboard logging is not complete, we take the minimum length between the two buffers
                min_length = min(len(eureka_rewards), len(oracle_rewards))
                rewards_correlation = np.corrcoef(eureka_rewards[:min_length], oracle_rewards[:min_length])[0, 1]

        success_metric_max = None
        # Make a summary of each plot in the tensorboard logs
        total_feed_back_string = ""
        for metric_name, metric_data in data.items():
            if "Eureka/" in metric_name:
                # Remove the first two data points as they are usually outliers
                metric_data = metric_data[2:]
                metric_name = metric_name.split("Eureka/", 1)[-1]
                metric_min = min(metric_data)
                metric_max = max(metric_data)
                metric_mean = sum(metric_data) / len(metric_data)
                # Best metric is the one closest to the target
                metric_best = metric_data[np.abs(np.array(metric_data) - self._success_metric_to_win).argmin()]
                if metric_name == "success_metric":
                    metric_name = "task_score"
                    success_metric_max = metric_best
                data_string = [f"{data:.2f}" for data in metric_data[::feedback_subsampling]]
                feedback_string = (
                    f"{metric_name}: {data_string}, Min: {metric_min:.2f}, Max: {metric_max:.2f}, Mean:"
                    f" {metric_mean:.2f} \n"
                )
                if "Eureka/success_metric" in data and metric_name == "Eureka/oracle_total_rewards":
                    # If success metric is available, we do not provide the oracle feedback
                    feedback_string = ""
                total_feed_back_string += feedback_string

        total_feed_back_string += f"\nThe desired task_score to win is: {self._success_metric_to_win:.2f}\n"
        return total_feed_back_string, success_metric_max, rewards_correlation

    def _log_iteration_results(self, iter: int, results: list):
        """Log the results of the iteration."""
        for idx, result in enumerate(results):
            print(f"{'*' * 20} Iteration {iter} / Process: {idx} {'*' * 20}")
            if result["success"]:
                print(f"Training successful with the following metrics:\n{result['eureka_task_feedback']}")
                print(f"Reward correlation with oracle rewards: {result['rewards_correlation']}")
            else:
                print(f"Training failed with the following exception:\n{result['exception']}\n")

        # write the iterations results to file
        with open(f"{self._log_dir}/eureka_iterations.txt", "a") as f:
            for idx, result in enumerate(results):
                f.write(f"{'#' * 20} Iteration: {iter} {'#' * 20}\n\n")
                f.write(f"{'*' * 20} Run: {idx} {'*' * 20}\n")
                f.write(f"- GPT reward method {result['assistant_prompt']}\n")
                f.write(f"- Feature idx: {result['feature_idx']}\n")
                if result["success"]:
                    f.write(f"Training successful with the following metrics:\n{result['eureka_task_feedback']}\n")
                    f.write(f"Reward correlation with oracle rewards:\n{result['rewards_correlation']}\n")
                    # Log success_metric, using 0.0 if it's None (e.g., if metric wasn't found in logs)
                    success_metric_value = result.get("success_metric_max")
                    if success_metric_value is None:
                        success_metric_value = 0.0
                    self._tensorboard_writer.add_scalar(f"Run_{idx}/success_metric", success_metric_value, iter)
                    # Log to wandb
                    if self._use_wandb and self._wandb:
                        self._wandb.log({
                            f"Run_{idx}/success_metric": success_metric_value,
                            f"Run_{idx}/rewards_correlation": result.get("rewards_correlation", 0.0),
                        }, step=iter)
                else:
                    f.write(f"Training failed with the following exception:\n{result['exception']}\n")
                    self._tensorboard_writer.add_scalar(f"Run_{idx}/success_metric", 0.0, iter)
                    # Log to wandb
                    if self._use_wandb and self._wandb:
                        self._wandb.log({f"Run_{idx}/success_metric": 0.0}, step=iter)
                self._tensorboard_writer.add_text(f"Run_{idx}/run_feedback", result["user_prompt"], iter)
                if self._use_wandb and self._wandb:
                    self._wandb.log({f"Run_{idx}/run_feedback": result["user_prompt"]}, step=iter)
                f.write("\n")

    def _log_final_results(self, best_run_results: dict):
        """Log the final results of the Eureka run."""
        output = ""
        if best_run_results["success_metric"] is not None:
            output += f"- Success metric: {best_run_results['success_metric']}\n"
            output += f"- GPT reward method: {best_run_results['gpt_reward_method']}\n"
            output += f"- Task metrics:\n{best_run_results['task_feedback']}\n"
            
            # Log final results to wandb
            if self._use_wandb and self._wandb:
                self._wandb.log({
                    "final/best_success_metric": best_run_results["success_metric"],
                    "final/gpt_reward_method": best_run_results["gpt_reward_method"],
                    "final/task_feedback": best_run_results["task_feedback"],
                })
        else:
            output += "- No successful training run\n"
            # Log to wandb
            if self._use_wandb and self._wandb:
                self._wandb.log({"final/best_success_metric": None})

        print("Final results:\n", output)

        with open(f"{self._log_dir}/eureka_final_result.txt", "w") as f:
            f.write(output)
        
        # Finish wandb run
        if self._use_wandb and self._wandb:
            self._wandb.finish()