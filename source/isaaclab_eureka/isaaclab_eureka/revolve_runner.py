# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

import datetime
import math
import os
import time
from typing import Dict, List, Optional

from isaaclab_eureka import EUREKA_ROOT_DIR
from isaaclab_eureka.config import (
    DIRECT_WORKFLOW_INITIAL_PROMPT,
    DIRECT_WORKFLOW_TASK_PROMPT,
    TASK_FAILURE_FEEDBACK_PROMPT,
    TASK_SUCCESS_POST_FEEDBACK_PROMPT,
    TASK_SUCCESS_PRE_FEEDBACK_PROMPT,
    TASKS_CFG,
)
from isaaclab_eureka.managers import EurekaTaskManager, LLMManager
from isaaclab_eureka.revolve import EloRanker, pairwise_preferences_from_metrics
from isaaclab_eureka.utils import load_tensorboard_logs


class Revolve:
    """Pairwise comparison baseline inspired by REvolve, reusing the Eureka stack."""

    def __init__(
        self,
        task: str,
        device: str = "cuda",
        env_seed: int = 42,
        rl_library: str = "rsl_rl",
        max_training_iterations: int = 100,
        feedback_subsampling: int = 10,
        temperature: float = 1.0,
        gpt_model: str = "gpt-4",
        num_pairs: int = 1,
        use_wandb: bool = True,
        wandb_project: str = "isaaclab-revolve",
        wandb_entity: str = None,
        wandb_name: str = None,
    ):
        if task not in TASKS_CFG:
            raise ValueError(
                f"Task configuration for {task} not found in the `TASKS_CFG` dictionary in config/tasks.py."
            )
        task_cfg = TASKS_CFG[task]
        self._task_description = task_cfg["description"]
        self._success_metric_to_win = task_cfg.get("success_metric_to_win")
        self._success_metric_tolerance = task_cfg.get("success_metric_tolerance")
        self._feedback_subsampling = feedback_subsampling
        # enforce even number of suggestions
        self._num_pairs = max(1, num_pairs)
        self._num_processes = self._num_pairs * 2

        print("[INFO]: Setting up the LLM Manager (revolve baseline)...")
        self._llm_manager = LLMManager(
            gpt_model=gpt_model,
            num_suggestions=self._num_processes,
            temperature=temperature,
            system_prompt=DIRECT_WORKFLOW_INITIAL_PROMPT,
        )

        print("[INFO]: Setting up the Task Manager (revolve baseline)...")
        self._task_manager = EurekaTaskManager(
            task=task,
            device=device,
            env_seed=env_seed,
            rl_library=rl_library,
            num_processes=self._num_processes,
            max_training_iterations=max_training_iterations,
            success_metric_string=task_cfg.get("success_metric"),
        )

        # Logging
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._log_dir = os.path.join(EUREKA_ROOT_DIR, "logs", "revolve", task, timestamp)
        os.makedirs(self._log_dir)

        from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

        self._tensorboard_writer = TensorboardSummaryWriter(log_dir=self._log_dir, flush_secs=10)

        self._use_wandb = use_wandb
        self._wandb = None
        if use_wandb:
            try:
                import wandb

                self._wandb = wandb
                run_name = wandb_name if wandb_name else f"{task}_{timestamp}"
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
                        "num_pairs": self._num_pairs,
                        "task_description": self._task_description,
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

        self._elo_ranker = EloRanker()

    def run(self, max_revolve_iterations: int):
        """Run revolve pairwise iterations."""
        import numpy as np

        user_prompt = DIRECT_WORKFLOW_TASK_PROMPT.format(
            task_description=self._task_description,
            success_metric_to_win=self._success_metric_to_win,
            get_observations_method_as_string=self._task_manager.get_observations_method_as_string,
        )
        assistant_prompt: Optional[str] = None
        best_run_results: Dict[str, Optional[float]] = {"success_metric": None}

        for iteration in range(max_revolve_iterations):
            print(f"\n{'#' * 20} Running REvolve Iteration {iteration} {'#' * 20} \n")
            llm_outputs = self._llm_manager.prompt(user_prompt=user_prompt, assistant_prompt=assistant_prompt)
            reward_strings = llm_outputs["reward_strings"]
            results = self._task_manager.train(reward_strings)
            time.sleep(1.0)  # allow tensorboard to flush

            for idx, result in enumerate(results):
                if not result["success"]:
                    user_feedback_prompt = TASK_FAILURE_FEEDBACK_PROMPT.format(traceback_msg=result["exception"])
                    result["eureka_task_feedback"] = ""
                    result["success_metric_max"] = None
                    result["rewards_correlation"] = 0.0
                else:
                    feedback, success_metric_max, rewards_correlation = self._get_eureka_task_feedback(
                        result["log_dir"], self._feedback_subsampling
                    )
                    user_feedback_prompt = (
                        TASK_SUCCESS_PRE_FEEDBACK_PROMPT.format(feedback_subsampling=self._feedback_subsampling)
                        + feedback
                        + TASK_SUCCESS_POST_FEEDBACK_PROMPT
                    )
                    result["eureka_task_feedback"] = feedback
                    result["success_metric_max"] = success_metric_max
                    result["rewards_correlation"] = rewards_correlation
                    if self._use_wandb and self._wandb:
                        self._wandb.log(
                            {
                                f"Run_{idx}/best_success_metric": success_metric_max if success_metric_max else 0.0,
                                f"Run_{idx}/rewards_correlation": rewards_correlation,
                            },
                            step=iteration,
                        )

                result["user_prompt"] = user_feedback_prompt
                result["assistant_prompt"] = llm_outputs["raw_outputs"][idx]

            pair_result = self._score_pairs(results, iteration)
            best_run_idx = pair_result["best_run_idx"]
            best_metric = results[best_run_idx].get("success_metric_max")
            if best_metric is not None and (
                best_run_results["success_metric"] is None
                or math.fabs(best_metric - self._success_metric_to_win)
                < math.fabs(best_run_results["success_metric"] - self._success_metric_to_win)
            ):
                best_run_results["success_metric"] = best_metric
                best_run_results["gpt_reward_method"] = reward_strings[best_run_idx]
                best_run_results["task_feedback"] = results[best_run_idx].get("eureka_task_feedback", "")
                if self._use_wandb and self._wandb:
                    self._wandb.log(
                        {
                            "best/overall_success_metric": best_metric,
                            "best/iteration": iteration,
                            "best/run_idx": best_run_idx,
                        },
                        step=iteration,
                    )

            self._log_iteration_results(iteration, results, pair_result)
            if (
                best_run_results["success_metric"] is not None
                and math.fabs(best_run_results["success_metric"] - self._success_metric_to_win)
                < self._success_metric_tolerance
            ):
                print(f"Task solved with success metric: {best_run_results['success_metric']}")
                break

            assistant_prompt = results[best_run_idx]["assistant_prompt"]
            user_prompt = results[best_run_idx]["user_prompt"]

        self._log_final_results(best_run_results)
        self._task_manager.close()

    def _score_pairs(self, results: List[Dict], iteration: int) -> Dict:
        """Compute pairwise winners and Elo ratings for this batch."""
        scores: List[tuple[str, float]] = []
        run_key_to_idx: Dict[str, int] = {}
        for idx, result in enumerate(results):
            run_key = f"iter{iteration}_run{idx}"
            score = result.get("success_metric_max")
            scores.append((run_key, 0.0 if score is None else score))
            run_key_to_idx[run_key] = idx

        matches = pairwise_preferences_from_metrics(
            scores, target=self._success_metric_to_win, higher_is_better=True
        )
        for item_a, item_b, winner in matches:
            self._elo_ranker.record_match(item_a, item_b, winner)

        ratings = self._elo_ranker.normalized_ratings()
        best_run_idx = 0
        if ratings:
            current_ratings = {k: v for k, v in ratings.items() if k in run_key_to_idx}
            if current_ratings:
                best_key = max(current_ratings, key=current_ratings.get)
                best_run_idx = run_key_to_idx.get(best_key, 0)

        return {"matches": matches, "ratings": ratings, "best_run_idx": best_run_idx}

    def _get_eureka_task_feedback(self, log_dir: str, feedback_subsampling: int):
        """Reuse Eureka feedback computation on tensorboard logs."""
        import numpy as np

        data = load_tensorboard_logs(log_dir)
        eureka_rewards_data = next((data[key] for key in data if key.endswith("Eureka/eureka_total_rewards")), None)
        oracle_rewards_data = next((data[key] for key in data if key.endswith("Eureka/oracle_total_rewards")), None)
        if eureka_rewards_data is None or oracle_rewards_data is None:
            rewards_correlation = 0.0
        else:
            eureka_rewards = np.array(eureka_rewards_data)
            oracle_rewards = np.array(oracle_rewards_data)
            if eureka_rewards.ndim == 0 or oracle_rewards.ndim == 0:
                rewards_correlation = 0.0
            elif len(eureka_rewards) == 0 or len(oracle_rewards) == 0:
                rewards_correlation = 0.0
            else:
                min_length = min(len(eureka_rewards), len(oracle_rewards))
                rewards_correlation = np.corrcoef(eureka_rewards[:min_length], oracle_rewards[:min_length])[0, 1]

        success_metric_max = None
        total_feed_back_string = ""
        for metric_name, metric_data in data.items():
            if "Eureka/" in metric_name:
                metric_data = metric_data[2:]
                metric_name = metric_name.split("Eureka/", 1)[-1]
                metric_min = min(metric_data) if metric_data else 0.0
                metric_max = max(metric_data) if metric_data else 0.0
                metric_mean = sum(metric_data) / len(metric_data) if metric_data else 0.0
                metric_best = 0.0
                if metric_data:
                    metric_best = metric_data[
                        math.fabs(np.array(metric_data) - self._success_metric_to_win).argmin()
                    ]
                if metric_name == "success_metric":
                    metric_name = "task_score"
                    success_metric_max = metric_best
                data_string = [f"{data:.2f}" for data in metric_data[::feedback_subsampling]]
                feedback_string = (
                    f"{metric_name}: {data_string}, Min: {metric_min:.2f}, Max: {metric_max:.2f}, Mean:"
                    f" {metric_mean:.2f} \n"
                )
                if "Eureka/success_metric" in data and metric_name == "Eureka/oracle_total_rewards":
                    feedback_string = ""
                total_feed_back_string += feedback_string
        total_feed_back_string += f"\nThe desired task_score to win is: {self._success_metric_to_win:.2f}\n"
        return total_feed_back_string, success_metric_max, rewards_correlation

    def _log_iteration_results(self, iteration: int, results: List[Dict], pair_result: Dict) -> None:
        """Log per-iteration outcomes."""
        for idx, result in enumerate(results):
            print(f"{'*' * 20} Iteration {iteration} / Process: {idx} {'*' * 20}")
            if result["success"]:
                print(f"Training successful with the following metrics:\n{result['eureka_task_feedback']}")
                print(f"Reward correlation with oracle rewards: {result['rewards_correlation']}")
            else:
                print(f"Training failed with the following exception:\n{result['exception']}\n")

        with open(f"{self._log_dir}/revolve_iterations.txt", "a") as f:
            for idx, result in enumerate(results):
                f.write(f"{'#' * 20} Iteration: {iteration} {'#' * 20}\n\n")
                f.write(f"{'*' * 20} Run: {idx} {'*' * 20}\n")
                f.write(f"- GPT reward method {result['assistant_prompt']}\n")
                if result["success"]:
                    f.write(f"Training successful with the following metrics:\n{result['eureka_task_feedback']}\n")
                    f.write(f"Reward correlation with oracle rewards:\n{result['rewards_correlation']}\n")
                    success_metric_value = result.get("success_metric_max") or 0.0
                    self._tensorboard_writer.add_scalar(f"Run_{idx}/success_metric", success_metric_value, iteration)
                    if self._use_wandb and self._wandb:
                        self._wandb.log(
                            {
                                f"Run_{idx}/success_metric": success_metric_value,
                                f"Run_{idx}/rewards_correlation": result.get("rewards_correlation", 0.0),
                            },
                            step=iteration,
                        )
                else:
                    f.write(f"Training failed with the following exception:\n{result['exception']}\n")
                    self._tensorboard_writer.add_scalar(f"Run_{idx}/success_metric", 0.0, iteration)
                    if self._use_wandb and self._wandb:
                        self._wandb.log({f"Run_{idx}/success_metric": 0.0}, step=iteration)
                self._tensorboard_writer.add_text(f"Run_{idx}/run_feedback", result["user_prompt"], iteration)
                if self._use_wandb and self._wandb:
                    self._wandb.log({f"Run_{idx}/run_feedback": result["user_prompt"]}, step=iteration)
                f.write("\n")

            f.write("Pairwise matches:\n")
            for item_a, item_b, winner in pair_result["matches"]:
                f.write(f"- {item_a} vs {item_b} -> {winner if winner else 'tie'}\n")
            f.write(f"Ratings: {pair_result['ratings']}\n\n")

    def _log_final_results(self, best_run_results: Dict) -> None:
        """Log the final best reward and rating."""
        output = ""
        if best_run_results.get("success_metric") is not None:
            output += f"- Success metric: {best_run_results['success_metric']}\n"
            output += f"- GPT reward method: {best_run_results.get('gpt_reward_method')}\n"
            output += f"- Task metrics:\n{best_run_results.get('task_feedback', '')}\n"
            if self._use_wandb and self._wandb:
                self._wandb.log(
                    {
                        "final/best_success_metric": best_run_results["success_metric"],
                        "final/gpt_reward_method": best_run_results.get("gpt_reward_method"),
                        "final/task_feedback": best_run_results.get("task_feedback", ""),
                    }
                )
        else:
            output += "- No successful training run\n"
            if self._use_wandb and self._wandb:
                self._wandb.log({"final/best_success_metric": None})

        print("Final results:\n", output)
        with open(f"{self._log_dir}/revolve_final_result.txt", "w") as f:
            f.write(output)
        if self._use_wandb and self._wandb:
            self._wandb.finish()
