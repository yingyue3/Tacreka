# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

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
    TASK_FAILURE_FEEDBACK_PROMPT,
    TASK_SUCCESS_POST_FEEDBACK_PROMPT,
    TASK_SUCCESS_PRE_FEEDBACK_PROMPT,
    TASKS_CFG,
)
from isaaclab_eureka.managers import EurekaTaskManager, LLMManager, ManipulationTaskManager
from isaaclab_eureka.learning_curve_utils import export_learning_curve_artifacts, resolve_checkpoint_path
from isaaclab_eureka.utils import summarize_tensorboard_candidate


class Eureka:
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
        num_reward_seeds: int = 5,
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
        self._num_reward_seeds = num_reward_seeds

        # Logging
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._log_dir = os.path.join(EUREKA_ROOT_DIR, "logs", "eureka", task, timestamp)
        self._rl_runs_dir = os.path.join(self._log_dir, "rl_runs")
        os.makedirs(self._log_dir)

        print("[INFO]: Setting up the LLM Manager...")
        self._llm_manager = LLMManager(
            gpt_model=gpt_model,
            num_suggestions=self._num_processes,
            temperature=temperature,
            system_prompt=DIRECT_WORKFLOW_INITIAL_PROMPT,
        )

        print("[INFO]: Setting up the Task Manager...")
        if task == "Isaac-Lift-Cube-Franka-v0":
            self._task_manager = ManipulationTaskManager(
                task=task,
                device=device,
                env_seed=env_seed,
                rl_library=rl_library,
                # num_processes=self._num_processes,
                num_processes=1,
                max_training_iterations=max_training_iterations,
                # success_metric_string=success_metric_string,
                log_namespace="tacreka_sr",
                rl_log_root_dir=self._rl_runs_dir,
            )
        else:
            self._task_manager = EurekaTaskManager(
                task=task,
                device=device,
                env_seed=env_seed,
                rl_library=rl_library,
                num_processes=1,
                max_training_iterations=max_training_iterations,
                success_metric_string=success_metric_string,
                log_namespace="tacreka_sr",
                rl_log_root_dir=self._rl_runs_dir,
            )

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

    def run(self, max_eureka_iterations: int):
        """Run the Eureka training loop.

        Args:
            max_eureka_iterations: The maximum number of Eureka iterations to run.
        """
        # We import here because doing this before launching Kit causes GCC_12.0 errors
        import numpy as np

        # Initial prompts
        user_prompt = DIRECT_WORKFLOW_TASK_PROMPT.format(
            task_description=self._task_description,
            success_metric_to_win=self._success_metric_to_win,
            get_observations_method_as_string=self._task_manager.get_observations_method_as_string,
        )
        # The assistant prompt is used to feed the previous LLM output back into the LLM
        assistant_prompt = None

        # The best run across all iterations
        best_run_results = {"success_metric": None}
        
        for iter in range(max_eureka_iterations):
            results = []

            print(f"\n{'#' * 20} Running Eureka Iteration {iter} {'#' * 20} \n")
            # Generate the GPT reward methods
            llm_outputs = self._llm_manager.prompt(user_prompt=user_prompt, assistant_prompt=assistant_prompt)
            gpt_reward_method_strings = llm_outputs["reward_strings"]
            # Log the llm outputs
            for idx, gpt_reward_method_string in enumerate(gpt_reward_method_strings):
                self._tensorboard_writer.add_text(f"Run_{idx}/raw_llm_output", llm_outputs["raw_outputs"][idx], iter)
                if self._use_wandb and self._wandb:
                    self._wandb.log({f"Run_{idx}/raw_llm_output": llm_outputs["raw_outputs"][idx]}, step=iter)
            # Train the RL agent
                results += self._task_manager.train([gpt_reward_method_string])
            # results = self._task_manager.train(gpt_reward_method_strings)
            # Give TensorBoard time to flush logs before reading them
            import time
            time.sleep(1.0)  # Wait 1 second for TensorBoard to flush
            # Evaluate the results
            iter_best_success_metric = None
            best_run_idx = 0
            for idx, result in enumerate(results):
                print(f"\n{'+' * 20} Evaluating Eureka Result {idx} {'+' * 20} \n")
                if not result["success"]:
                    user_feedback_prompt = TASK_FAILURE_FEEDBACK_PROMPT.format(traceback_msg=result["exception"])
                else:
                    evaluation_summary = self._get_eureka_task_feedback(
                        result.get("seed_log_dirs") or result["log_dir"], self._feedback_subsampling
                    )
                    eureka_task_feedback = evaluation_summary["feedback"]
                    success_metric_mean = evaluation_summary["success_metric_mean"]
                    rewards_correlation_mean = evaluation_summary["rewards_correlation_mean"]
                    best_seed_index = evaluation_summary["best_seed_index"]
                    representative_seed_index = evaluation_summary["representative_seed_index"]
                    seed_results = result.get("seed_results", [])
                    seed_run_dirs = result.get("seed_run_dirs", [])
                    seed_checkpoint_files = result.get("seed_checkpoint_files", [])
                    selected_seed_result = seed_results[best_seed_index] if best_seed_index < len(seed_results) else {}
                    representative_seed_result = (
                        seed_results[representative_seed_index]
                        if representative_seed_index < len(seed_results)
                        else {}
                    )

                    # Generate the user feedback prompt
                    user_feedback_prompt = (
                        TASK_SUCCESS_PRE_FEEDBACK_PROMPT.format(feedback_subsampling=self._feedback_subsampling)
                        + eureka_task_feedback
                        + TASK_SUCCESS_POST_FEEDBACK_PROMPT
                    )

                    # Store the results
                    results[idx]["eureka_task_feedback"] = eureka_task_feedback
                    results[idx]["success_metric_mean"] = success_metric_mean
                    results[idx]["success_metric_max"] = success_metric_mean
                    results[idx]["success_metric_stderr"] = evaluation_summary["success_metric_stderr"]
                    results[idx]["rewards_correlation"] = rewards_correlation_mean
                    results[idx]["rewards_correlation_stderr"] = evaluation_summary["rewards_correlation_stderr"]
                    results[idx]["seed_summaries"] = evaluation_summary["seed_summaries"]
                    results[idx]["selected_seed"] = selected_seed_result.get("seed", best_seed_index)
                    results[idx]["representative_seed"] = representative_seed_result.get(
                        "seed", representative_seed_index
                    )
                    results[idx]["selected_seed_log_dir"] = selected_seed_result.get("log_dir")
                    results[idx]["selected_seed_run_dir"] = selected_seed_result.get(
                        "run_dir",
                        seed_run_dirs[best_seed_index] if best_seed_index < len(seed_run_dirs) else None,
                    )
                    results[idx]["selected_seed_checkpoint_file"] = selected_seed_result.get(
                        "checkpoint_file",
                        seed_checkpoint_files[best_seed_index] if best_seed_index < len(seed_checkpoint_files) else None,
                    )
                    # Log metrics to wandb
                    if self._use_wandb and self._wandb:
                        self._wandb.log({
                            f"Run_{idx}/best_success_metric": success_metric_mean if success_metric_mean is not None else 0.0,
                            f"Run_{idx}/success_metric_stderr": evaluation_summary["success_metric_stderr"] or 0.0,
                            f"Run_{idx}/rewards_correlation": rewards_correlation_mean or 0.0,
                        }, step=iter)
                    # Check the best performing metric, determined by the minimum distance from the win target
                    if success_metric_mean is not None and (
                        iter_best_success_metric is None
                        or np.abs(success_metric_mean - self._success_metric_to_win)
                        < np.abs(iter_best_success_metric - self._success_metric_to_win)
                    ):
                        # Store the best run for this iteration
                        iter_best_success_metric = success_metric_mean
                        best_run_idx = idx

                        # Store the best metric across all iterations
                        if best_run_results["success_metric"] is None or (
                            np.abs(iter_best_success_metric - self._success_metric_to_win)
                            < np.abs(best_run_results["success_metric"] - self._success_metric_to_win)
                        ):
                            best_run_results["success_metric"] = iter_best_success_metric
                            best_run_results["success_metric_stderr"] = evaluation_summary["success_metric_stderr"]
                            best_run_results["gpt_reward_method"] = gpt_reward_method_strings[idx]
                            best_run_results["task_feedback"] = eureka_task_feedback
                            best_run_results["rewards_correlation"] = rewards_correlation_mean
                            best_run_results["rewards_correlation_stderr"] = evaluation_summary[
                                "rewards_correlation_stderr"
                            ]
                            best_run_results["training_log_dir"] = result.get("seed_log_dirs") or result.get("log_dir")
                            best_run_results["training_log_dirs"] = result.get("seed_log_dirs") or [result.get("log_dir")]
                            best_run_results["training_run_dir"] = (
                                results[idx]["selected_seed_run_dir"]
                                or result.get("run_dir", result.get("log_dir"))
                            )
                            best_run_results["checkpoint_file"] = (
                                results[idx]["selected_seed_checkpoint_file"]
                                or resolve_checkpoint_path(results[idx]["selected_seed_run_dir"])
                            )
                            best_run_results["learning_curve"] = result.get("learning_curve")
                            best_run_results["seed_summaries"] = evaluation_summary["seed_summaries"]
                            best_run_results["selected_seed"] = results[idx]["selected_seed"]
                            best_run_results["representative_seed"] = results[idx]["representative_seed"]
                            # Log best metric to wandb
                            if self._use_wandb and self._wandb:
                                self._wandb.log({
                                    "best/overall_success_metric": iter_best_success_metric,
                                    "best/iteration": iter,
                                    "best/run_idx": idx,
                                }, step=iter)

                # Add the prompts
                results[idx]["user_prompt"] = user_feedback_prompt
                results[idx]["assistant_prompt"] = llm_outputs["raw_outputs"][idx]

            self._log_iteration_results(iter, results)

            # Disabled early stopping on hitting the success metric target so the
            # runner always uses the full outer-iteration budget.
            # if (
            #     best_run_results["success_metric"] is not None
            #     and np.abs(best_run_results["success_metric"] - self._success_metric_to_win)
            #     < self._success_metric_tolerance
            # ):
            #     print(f"Task solved with success metric: {best_run_results['success_metric']}")
            #     break

            assistant_prompt = results[best_run_idx]["assistant_prompt"]
            user_prompt = results[best_run_idx]["user_prompt"]

        self._log_final_results(best_run_results)
        # Close the task manager
        self._task_manager.close()

    def _get_eureka_task_feedback(self, log_dir: str | list[str], feedback_subsampling: int) -> dict:
        """Aggregate one candidate across one or more seed log directories."""
        return summarize_tensorboard_candidate(
            log_dirs=log_dir,
            feedback_subsampling=feedback_subsampling,
            success_metric_target=self._success_metric_to_win,
        )

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
                if result["success"]:
                    f.write(f"Training successful with the following metrics:\n{result['eureka_task_feedback']}\n")
                    f.write(f"Reward correlation with oracle rewards:\n{result['rewards_correlation']}\n")
                    # Log success_metric, using 0.0 if it's None (e.g., if metric wasn't found in logs)
                    success_metric_value = result.get("success_metric_mean")
                    if success_metric_value is None:
                        success_metric_value = 0.0
                    self._tensorboard_writer.add_scalar(f"Run_{idx}/success_metric", success_metric_value, iter)
                    success_metric_stderr = result.get("success_metric_stderr") or 0.0
                    self._tensorboard_writer.add_scalar(f"Run_{idx}/success_metric_stderr", success_metric_stderr, iter)
                    # Log to wandb
                    if self._use_wandb and self._wandb:
                        self._wandb.log({
                            f"Run_{idx}/success_metric": success_metric_value,
                            f"Run_{idx}/success_metric_stderr": success_metric_stderr,
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
        if best_run_results.get("training_log_dir"):
            best_learning_curve = export_learning_curve_artifacts(
                best_run_results["training_log_dir"],
                output_dir=os.path.join(self._log_dir, "best_run_learning_curves"),
                run_name="best_run",
            )
            if best_learning_curve is not None:
                best_run_results["best_learning_curve"] = best_learning_curve

        output = ""
        if best_run_results["success_metric"] is not None:
            output += f"- Success metric: {best_run_results['success_metric']}\n"
            output += f"- Success metric stderr: {best_run_results.get('success_metric_stderr', 0.0)}\n"
            output += f"- GPT reward method: {best_run_results['gpt_reward_method']}\n"
            output += f"- Best training log dir: {best_run_results.get('training_log_dir', 'unknown')}\n"
            output += f"- Best training run dir: {best_run_results.get('training_run_dir', 'unknown')}\n"
            output += f"- Best checkpoint: {best_run_results.get('checkpoint_file', 'unknown')}\n"
            output += f"- Selected seed: {best_run_results.get('selected_seed', 'unknown')}\n"
            output += f"- Representative seed: {best_run_results.get('representative_seed', 'unknown')}\n"
            learning_curve_path = best_run_results.get("best_learning_curve", {}).get("plot_path", "unknown")
            output += f"- Best learning curve plot: {learning_curve_path}\n"
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

        with open(f"{self._log_dir}/best_run.json", "w") as f:
            json.dump(best_run_results, f, indent=2, default=str)
        
        # Finish wandb run
        if self._use_wandb and self._wandb:
            self._wandb.finish()
