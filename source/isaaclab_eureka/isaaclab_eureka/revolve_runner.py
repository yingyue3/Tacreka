# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

import datetime
import json
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
from isaaclab_eureka.learning_curve_utils import export_learning_curve_artifacts, resolve_checkpoint_path
from isaaclab_eureka.revolve import EloRanker, pairwise_preferences_from_metrics
from isaaclab_eureka.utils import summarize_tensorboard_candidate


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
        num_reward_seeds: int = 5,
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
        self._num_reward_seeds = num_reward_seeds
        # enforce even number of suggestions
        self._num_pairs = max(1, num_pairs)
        self._num_processes = self._num_pairs * 2

        # Logging
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._log_dir = os.path.join(EUREKA_ROOT_DIR, "logs", "revolve", task, timestamp)
        self._rl_runs_dir = os.path.join(self._log_dir, "rl_runs")
        os.makedirs(self._log_dir)

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
            log_namespace="revolve",
            rl_log_root_dir=self._rl_runs_dir,
            num_seeds_per_reward=num_reward_seeds,
        )

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
                    evaluation_summary = self._get_eureka_task_feedback(
                        result.get("seed_log_dirs") or result["log_dir"], self._feedback_subsampling
                    )
                    feedback = evaluation_summary["feedback"]
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
                    user_feedback_prompt = (
                        TASK_SUCCESS_PRE_FEEDBACK_PROMPT.format(feedback_subsampling=self._feedback_subsampling)
                        + feedback
                        + TASK_SUCCESS_POST_FEEDBACK_PROMPT
                    )
                    result["eureka_task_feedback"] = feedback
                    result["success_metric_mean"] = success_metric_mean
                    result["success_metric_max"] = success_metric_mean
                    result["success_metric_stderr"] = evaluation_summary["success_metric_stderr"]
                    result["rewards_correlation"] = rewards_correlation_mean
                    result["rewards_correlation_stderr"] = evaluation_summary["rewards_correlation_stderr"]
                    result["seed_summaries"] = evaluation_summary["seed_summaries"]
                    result["selected_seed"] = selected_seed_result.get("seed", best_seed_index)
                    result["representative_seed"] = representative_seed_result.get(
                        "seed", representative_seed_index
                    )
                    result["selected_seed_log_dir"] = selected_seed_result.get("log_dir")
                    result["selected_seed_run_dir"] = selected_seed_result.get(
                        "run_dir",
                        seed_run_dirs[best_seed_index] if best_seed_index < len(seed_run_dirs) else None,
                    )
                    result["selected_seed_checkpoint_file"] = selected_seed_result.get(
                        "checkpoint_file",
                        seed_checkpoint_files[best_seed_index] if best_seed_index < len(seed_checkpoint_files) else None,
                    )
                    if self._use_wandb and self._wandb:
                        self._wandb.log(
                            {
                                f"Run_{idx}/best_success_metric": success_metric_mean if success_metric_mean else 0.0,
                                f"Run_{idx}/success_metric_stderr": evaluation_summary["success_metric_stderr"] or 0.0,
                                f"Run_{idx}/rewards_correlation": rewards_correlation_mean or 0.0,
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
                best_run_results["success_metric_stderr"] = results[best_run_idx].get("success_metric_stderr")
                best_run_results["gpt_reward_method"] = reward_strings[best_run_idx]
                best_run_results["task_feedback"] = results[best_run_idx].get("eureka_task_feedback", "")
                best_run_results["rewards_correlation"] = results[best_run_idx].get("rewards_correlation")
                best_run_results["rewards_correlation_stderr"] = results[best_run_idx].get(
                    "rewards_correlation_stderr"
                )
                best_run_results["training_log_dir"] = results[best_run_idx].get("seed_log_dirs") or results[best_run_idx].get("log_dir")
                best_run_results["training_log_dirs"] = results[best_run_idx].get("seed_log_dirs") or [results[best_run_idx].get("log_dir")]
                best_run_results["training_run_dir"] = (
                    results[best_run_idx].get("selected_seed_run_dir")
                    or results[best_run_idx].get("run_dir", results[best_run_idx].get("log_dir"))
                )
                best_run_results["checkpoint_file"] = (
                    results[best_run_idx].get("selected_seed_checkpoint_file")
                    or resolve_checkpoint_path(results[best_run_idx].get("selected_seed_run_dir"))
                )
                best_run_results["learning_curve"] = results[best_run_idx].get("learning_curve")
                best_run_results["seed_summaries"] = results[best_run_idx].get("seed_summaries")
                best_run_results["selected_seed"] = results[best_run_idx].get("selected_seed")
                best_run_results["representative_seed"] = results[best_run_idx].get("representative_seed")
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

    def _get_eureka_task_feedback(self, log_dir: str | list[str], feedback_subsampling: int) -> dict:
        """Reuse Eureka feedback computation on one or more tensorboard log directories."""
        return summarize_tensorboard_candidate(
            log_dirs=log_dir,
            feedback_subsampling=feedback_subsampling,
            success_metric_target=self._success_metric_to_win,
        )

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
                    success_metric_value = result.get("success_metric_mean") or 0.0
                    self._tensorboard_writer.add_scalar(f"Run_{idx}/success_metric", success_metric_value, iteration)
                    success_metric_stderr = result.get("success_metric_stderr") or 0.0
                    self._tensorboard_writer.add_scalar(f"Run_{idx}/success_metric_stderr", success_metric_stderr, iteration)
                    if self._use_wandb and self._wandb:
                        self._wandb.log(
                            {
                                f"Run_{idx}/success_metric": success_metric_value,
                                f"Run_{idx}/success_metric_stderr": success_metric_stderr,
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
        if best_run_results.get("training_log_dir"):
            best_learning_curve = export_learning_curve_artifacts(
                best_run_results["training_log_dir"],
                output_dir=os.path.join(self._log_dir, "best_run_learning_curves"),
                run_name="best_run",
            )
            if best_learning_curve is not None:
                best_run_results["best_learning_curve"] = best_learning_curve

        output = ""
        if best_run_results.get("success_metric") is not None:
            output += f"- Success metric: {best_run_results['success_metric']}\n"
            output += f"- Success metric stderr: {best_run_results.get('success_metric_stderr', 0.0)}\n"
            output += f"- GPT reward method: {best_run_results.get('gpt_reward_method')}\n"
            output += f"- Best training log dir: {best_run_results.get('training_log_dir', 'unknown')}\n"
            output += f"- Best training run dir: {best_run_results.get('training_run_dir', 'unknown')}\n"
            output += f"- Best checkpoint: {best_run_results.get('checkpoint_file', 'unknown')}\n"
            output += f"- Selected seed: {best_run_results.get('selected_seed', 'unknown')}\n"
            output += f"- Representative seed: {best_run_results.get('representative_seed', 'unknown')}\n"
            learning_curve_path = best_run_results.get("best_learning_curve", {}).get("plot_path", "unknown")
            output += f"- Best learning curve plot: {learning_curve_path}\n"
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
        with open(f"{self._log_dir}/best_run.json", "w") as f:
            json.dump(best_run_results, f, indent=2, default=str)
        if self._use_wandb and self._wandb:
            self._wandb.finish()
