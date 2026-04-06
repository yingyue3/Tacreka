# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

import datetime
import glob
import json
import math
import os
import random
from functools import partial
from typing import Dict, List, Optional, Tuple

from isaaclab_eureka import EUREKA_ROOT_DIR
from isaaclab_eureka.config import (
    DIRECT_WORKFLOW_INITIAL_PROMPT,
    DIRECT_WORKFLOW_TASK_PROMPT,
    TASKS_CFG,
)
from isaaclab_eureka.managers import EurekaTaskManager, LLMManager
from isaaclab_eureka.learning_curve_utils import export_learning_curve_artifacts
from isaaclab_eureka.revolve_full.database import RevolveDatabase
from isaaclab_eureka.revolve_full import prompts as revolve_prompts
from isaaclab_eureka.revolve_full.human_feedback import compute_hf_scores
from isaaclab_eureka.utils import summarize_tensorboard_candidate


def _linear_decay(iteration: int, initial: float, final: float, num_iterations: int):
    return initial - (initial - final) * iteration / max(num_iterations, 1)


def _append_in_context_prompt(
    task_prompt: str, samples: List[Tuple[str, float]], operator: str, episodes: int
) -> str:
    """Augment the task prompt with mutation/crossover guidance and few-shot examples."""
    if operator == "crossover":
        template = revolve_prompts.CROSSOVER
    else:
        template = revolve_prompts.MUTATION

    examples_lines: List[str] = []
    if not samples:
        examples_lines.append("No prior examples available.")
    else:
        for filename, fitness_score in samples:
            try:
                with open(filename, "r") as f:
                    fn_str = f.read()
                examples_lines.append(f"\nscore={fitness_score:.3f}\n```python\n{fn_str}\n```")
            except FileNotFoundError:
                continue
    examples_block = "\n".join(examples_lines)
    template = template.replace("<EXAMPLES>", examples_block)
    template = template.replace("<EPISODES>", str(episodes))
    task_rules = (
        "\nPlease keep the Isaac Lab reward signature `_get_rewards_eureka(self)` and return "
        "a tuple of (total_reward, reward_dict) where reward_dict maps strings to tensors on the correct device."
    )
    return task_prompt + task_rules + "\n" + template


def generate_valid_reward(llm_manager, user_prompt: str, max_trials: int = 3) -> Optional[str]:
    """Loop until we get a reward string that looks usable."""
    error_feedback = ""
    trials = 0
    while trials < max_trials:
        llm_outputs = llm_manager.prompt(user_prompt=user_prompt + error_feedback)
        reward_string = llm_outputs["reward_strings"][0]
        if reward_string and "_get_rewards_eureka" in reward_string and "return" in reward_string:
            return reward_string
        error_feedback = (
            "\nThe previous reward function was invalid (missing `_get_rewards_eureka` or `return`). "
            "Please fix and regenerate a valid reward function."
        )
        trials += 1
    return None


class RevolveFull:
    """Island-based REvolve-style baseline adapted to Isaac Lab tasks."""

    def __init__(
        self,
        task: str,
        device: str = "cuda",
        env_seed: int = 42,
        rl_library: str = "rsl_rl",
        max_training_iterations: int = 100,
        temperature: float = 1.0,
        gpt_model: str = "gpt-4",
        num_generations: int = 5,
        individuals_per_generation: int = 6,
        num_islands: int = 4,
        max_island_size: int = 8,
        crossover_prob: float = 0.5,
        migration_prob: float = 0.3,
        num_reward_seeds: int = 5,
        few_shot: Optional[Dict[str, int]] = None,
        temperature_final: float = 1.0,
        use_human_feedback: bool = False,
        human_feedback_dir: Optional[str] = None,
        use_wandb: bool = True,
        wandb_project: str = "isaaclab-revolve-full",
        wandb_entity: str = None,
        wandb_name: str = None,
    ):
        if task not in TASKS_CFG:
            raise ValueError(
                f"Task configuration for {task} not found in the `TASKS_CFG` dictionary in config/tasks.py."
            )
        self._task_cfg = TASKS_CFG[task]
        self._task = task
        self._num_generations = num_generations
        self._individuals_per_generation = individuals_per_generation
        self._few_shot = few_shot or {"mutation": 1, "crossover": 2}
        self._num_reward_seeds = num_reward_seeds
        self._temperature_initial = temperature
        self._temperature_final = temperature_final
        self._num_islands = num_islands
        self._max_island_size = max_island_size
        self._crossover_prob = crossover_prob
        self._migration_prob = migration_prob
        self._use_hf = use_human_feedback
        self._hf_dir = human_feedback_dir
        self._success_metric_target = float(self._task_cfg["success_metric_to_win"])
        # Island selection assumes "higher is better". We map to a distance-based fitness.
        self._failure_fitness = -1e9

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._log_dir = os.path.join(EUREKA_ROOT_DIR, "logs", "revolve_full", task, timestamp)
        self._rl_runs_dir = os.path.join(self._log_dir, "rl_runs")
        self._db_dir = os.path.join(self._log_dir, "database")
        os.makedirs(self._db_dir, exist_ok=True)
        if self._use_hf:
            self._hf_dir = human_feedback_dir or os.path.join(self._log_dir, "human_feedback")
            os.makedirs(self._hf_dir, exist_ok=True)

        self._llm_factory = partial(
            LLMManager,
            gpt_model=gpt_model,
            num_suggestions=1,
            temperature=temperature,
            system_prompt=DIRECT_WORKFLOW_INITIAL_PROMPT,
        )
        self._task_manager = EurekaTaskManager(
            task=task,
            device=device,
            env_seed=env_seed,
            rl_library=rl_library,
            num_processes=1,
            max_training_iterations=max_training_iterations,
            success_metric_string=self._task_cfg.get("success_metric"),
            log_namespace="revolve_full",
            rl_log_root_dir=self._rl_runs_dir,
            num_seeds_per_reward=num_reward_seeds,
        )

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
                        "temperature": temperature,
                        "gpt_model": gpt_model,
                        "num_generations": num_generations,
                        "individuals_per_generation": individuals_per_generation,
                        "num_islands": num_islands,
                        "max_island_size": max_island_size,
                        "crossover_prob": crossover_prob,
                        "migration_prob": migration_prob,
                        "fitness_formula": "negative_abs_distance_to_target",
                    },
                    dir=self._log_dir,
                )
            except ImportError:
                print("[WARNING]: wandb not installed. Install with 'pip install wandb' to enable wandb logging.")
                self._use_wandb = False
                self._wandb = None

    def run(self):
        best_overall = {"fitness": None, "reward": None, "feedback": "", "rewards_correlation": None}
        base_user_prompt = DIRECT_WORKFLOW_TASK_PROMPT.format(
            task_description=self._task_cfg["description"],
            success_metric_to_win=self._task_cfg["success_metric_to_win"],
            get_observations_method_as_string=self._task_manager.get_observations_method_as_string,
        )

        for generation_id in range(self._num_generations):
            temperature = _linear_decay(
                iteration=generation_id,
                initial=self._temperature_initial,
                final=self._temperature_final,
                num_iterations=self._num_generations,
            )
            print(
                f"\n========= Generation {generation_id} | temperature: {round(temperature, 2)} =========="
            )
            database = RevolveDatabase(
                num_islands=self._num_islands,
                max_size=self._max_island_size,
                crossover_prob=self._crossover_prob,
                migration_prob=self._migration_prob,
                load_islands=not generation_id == 0,
                reward_fn_dir=self._db_dir,
            )

            rew_fn_strings: List[str] = []
            fitness_scores: List[float] = []
            island_ids: List[int] = []
            counter_ids: List[int] = []
            metrics_dicts: List[Dict] = []
            candidate_ids: List[str] = []

            for counter_id in range(self._individuals_per_generation):
                if generation_id == 0:
                    island_id = random.choice(range(database.num_islands))
                    in_context_samples: List[Tuple[str, float]] = []
                    operator = "mutation"
                else:
                    in_context_samples, island_id, operator = database.sample_in_context(
                        self._few_shot, temperature
                    )
                island_ids.append(island_id)
                prompt_with_context = _append_in_context_prompt(
                    base_user_prompt, in_context_samples, operator, episodes=100
                )

                llm_manager = self._llm_factory(temperature=temperature)
                reward_string = generate_valid_reward(llm_manager, prompt_with_context)
                if reward_string is None or reward_string.strip() == "":
                    print(f"[WARN] Empty reward string for generation {generation_id}, counter {counter_id}. Skipping.")
                    continue

                result = self._task_manager.train([reward_string])[0]
                success_metric_value = None
                fitness = self._failure_fitness
                feedback = ""
                correlation = 0.0
                success_metric_stderr = None
                rewards_correlation_stderr = None
                selected_seed = None
                representative_seed = None
                if result["success"]:
                    evaluation_summary = self._get_eureka_task_feedback(
                        result.get("seed_log_dirs") or result["log_dir"], feedback_subsampling=10
                    )
                    feedback = evaluation_summary["feedback"]
                    success_metric_value = evaluation_summary["success_metric_mean"]
                    success_metric_stderr = evaluation_summary["success_metric_stderr"]
                    correlation = evaluation_summary["rewards_correlation_mean"]
                    rewards_correlation_stderr = evaluation_summary["rewards_correlation_stderr"]
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
                    selected_seed = selected_seed_result.get("seed", best_seed_index)
                    representative_seed = representative_seed_result.get("seed", representative_seed_index)
                    result["selected_seed_run_dir"] = selected_seed_result.get(
                        "run_dir",
                        seed_run_dirs[best_seed_index] if best_seed_index < len(seed_run_dirs) else None,
                    )
                    result["selected_seed_checkpoint_file"] = selected_seed_result.get(
                        "checkpoint_file",
                        seed_checkpoint_files[best_seed_index] if best_seed_index < len(seed_checkpoint_files) else None,
                    )
                    result["seed_summaries"] = evaluation_summary["seed_summaries"]
                    result["selected_seed"] = selected_seed
                    result["representative_seed"] = representative_seed
                    if success_metric_value is not None:
                        fitness = self._to_evolution_fitness(success_metric_value)
                else:
                    feedback = result.get("exception", "")
                    success_metric_value = None

                metrics_dict = {
                    "fitness": fitness,
                    "success_metric": success_metric_value,
                    "success_metric_stderr": success_metric_stderr,
                    "rewards_correlation": correlation,
                    "rewards_correlation_stderr": rewards_correlation_stderr,
                    "success": result["success"],
                    "operator": operator,
                    "selected_seed": selected_seed,
                    "representative_seed": representative_seed,
                }
                metrics_dicts.append(metrics_dict)
                rew_fn_strings.append(reward_string)
                fitness_scores.append(fitness)
                counter_ids.append(counter_id)
                candidate_ids.append(f"gen{generation_id}_ctr{counter_id}_isl{island_id}")

                if best_overall["fitness"] is None or (
                    success_metric_value is not None
                    and abs(success_metric_value - self._task_cfg["success_metric_to_win"])
                    < abs(best_overall["fitness"] - self._task_cfg["success_metric_to_win"])
                ):
                    candidate_id = f"gen{generation_id}_ctr{counter_id}_isl{island_id}"
                    reward_file = os.path.join(
                        self._db_dir, f"island_{island_id}", "generated_fns", f"{generation_id}_{counter_id}.txt"
                    )
                    fitness_file = os.path.join(
                        self._db_dir, f"island_{island_id}", "fitness_scores", f"{generation_id}_{counter_id}.txt"
                    )
                    best_overall["fitness"] = success_metric_value
                    best_overall["success_metric_stderr"] = success_metric_stderr
                    best_overall["reward"] = reward_string
                    best_overall["feedback"] = feedback
                    best_overall["rewards_correlation"] = correlation
                    best_overall["rewards_correlation_stderr"] = rewards_correlation_stderr
                    best_overall["evolution_fitness"] = fitness
                    best_overall["candidate_id"] = candidate_id
                    best_overall["candidate_generation"] = generation_id
                    best_overall["candidate_counter"] = counter_id
                    best_overall["candidate_island"] = island_id
                    best_overall["training_log_dir"] = result.get("seed_log_dirs") or result.get("log_dir")
                    best_overall["training_log_dirs"] = result.get("seed_log_dirs") or [result.get("log_dir")]
                    best_overall["training_run_dir"] = (
                        result.get("selected_seed_run_dir")
                        or result.get("run_dir", result.get("log_dir"))
                    )
                    best_overall["candidate_reward_file"] = reward_file
                    best_overall["candidate_fitness_file"] = fitness_file
                    best_overall["checkpoint_file"] = (
                        result.get("selected_seed_checkpoint_file")
                        or self._resolve_checkpoint_path(result.get("selected_seed_run_dir"))
                    )
                    best_overall["learning_curve"] = result.get("learning_curve")
                    best_overall["seed_summaries"] = result.get("seed_summaries")
                    best_overall["selected_seed"] = selected_seed
                    best_overall["representative_seed"] = representative_seed

            if len(rew_fn_strings) == 0:
                print("[WARN] No valid reward functions generated; skipping generation update.")
                continue

            # Human feedback: write manifest and, if responses exist, override fitness scores with Elo ratings.
            if self._use_hf:
                manifest_path = os.path.join(self._hf_dir, f"generation_{generation_id}", "candidates_manifest.csv")
                os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
                with open(manifest_path, "w") as mf:
                    mf.write("candidate_id,generation,counter,island,log_dir\n")
                    for cid, gen_id, ctr_id, isl_id in zip(
                        candidate_ids,
                        [generation_id] * len(counter_ids),
                        counter_ids,
                        island_ids,
                    ):
                        mf.write(f"{cid},{gen_id},{ctr_id},{isl_id},{self._log_dir}\n")
                hf_scores = compute_hf_scores(self._hf_dir, generation_id)
                if hf_scores:
                    remapped_scores = []
                    for cid, default_score in zip(candidate_ids, fitness_scores):
                        remapped_scores.append(hf_scores.get(cid, default_score))
                    fitness_scores = remapped_scores

            if generation_id > 0:
                database.add_individuals_to_islands(
                    [generation_id] * len(island_ids),
                    counter_ids,
                    rew_fn_strings,
                    fitness_scores,
                    metrics_dicts,
                    island_ids,
                )
            else:
                database.seed_islands(
                    [generation_id] * len(island_ids),
                    counter_ids,
                    rew_fn_strings,
                    fitness_scores,
                    metrics_dicts,
                    island_ids,
                )

        self._task_manager.close()
        self._log_final_results(best_overall)

    def _to_evolution_fitness(self, success_metric: float) -> float:
        """Map task score to island fitness where larger is always better."""
        return -abs(float(success_metric) - self._success_metric_target)

    @staticmethod
    def _resolve_checkpoint_path(log_dir: Optional[str]) -> Optional[str]:
        """Pick the latest model checkpoint from a training log directory."""
        if not log_dir:
            return None
        checkpoint_paths = glob.glob(os.path.join(log_dir, "model_*.pt"))
        if not checkpoint_paths:
            return None

        def _checkpoint_step(path: str) -> int:
            filename = os.path.basename(path)
            stem = os.path.splitext(filename)[0]
            try:
                return int(stem.split("_")[-1])
            except ValueError:
                return -1

        return max(checkpoint_paths, key=_checkpoint_step)

    def _get_eureka_task_feedback(self, log_dir: str | list[str], feedback_subsampling: int) -> dict:
        return summarize_tensorboard_candidate(
            log_dirs=log_dir,
            feedback_subsampling=feedback_subsampling,
            success_metric_target=self._task_cfg["success_metric_to_win"],
        )

    def _log_final_results(self, best_run_results: Dict) -> None:
        output = ""
        if best_run_results.get("training_log_dir"):
            best_learning_curve = export_learning_curve_artifacts(
                best_run_results["training_log_dir"],
                output_dir=os.path.join(self._log_dir, "best_run_learning_curves"),
                run_name="best_run",
            )
            if best_learning_curve is not None:
                best_run_results["best_learning_curve"] = best_learning_curve
        if best_run_results.get("fitness") is not None:
            output += f"- Success metric: {best_run_results['fitness']}\n"
            output += f"- Success metric stderr: {best_run_results.get('success_metric_stderr', 0.0)}\n"
            output += f"- Best candidate id: {best_run_results.get('candidate_id', 'unknown')}\n"
            output += (
                f"- Best candidate details: generation={best_run_results.get('candidate_generation', 'unknown')}, "
                f"counter={best_run_results.get('candidate_counter', 'unknown')}, "
                f"island={best_run_results.get('candidate_island', 'unknown')}\n"
            )
            output += (
                f"- Best candidate reward file: {best_run_results.get('candidate_reward_file', 'unknown')}\n"
            )
            output += (
                f"- Best candidate fitness file: {best_run_results.get('candidate_fitness_file', 'unknown')}\n"
            )
            output += (
                f"- Best candidate reward correlation: {best_run_results.get('rewards_correlation', 'unknown')}\n"
            )
            output += (
                f"- Best candidate reward correlation stderr: {best_run_results.get('rewards_correlation_stderr', 'unknown')}\n"
            )
            output += (
                f"- Evolution fitness formula: -abs(task_score - {self._success_metric_target:.4f})\n"
            )
            output += f"- Best candidate evolution fitness: {best_run_results.get('evolution_fitness', 'unknown')}\n"
            output += f"- Best candidate training log dir: {best_run_results.get('training_log_dir', 'unknown')}\n"
            output += f"- Best candidate training run dir: {best_run_results.get('training_run_dir', 'unknown')}\n"
            output += f"- Best candidate checkpoint: {best_run_results.get('checkpoint_file', 'unknown')}\n"
            output += f"- Selected seed: {best_run_results.get('selected_seed', 'unknown')}\n"
            output += f"- Representative seed: {best_run_results.get('representative_seed', 'unknown')}\n"
            learning_curve_path = best_run_results.get("best_learning_curve", {}).get("plot_path", "unknown")
            output += f"- Best candidate learning curve plot: {learning_curve_path}\n"
            output += f"- GPT reward method:\n{best_run_results.get('reward')}\n"
            output += f"- Task metrics:\n{best_run_results.get('feedback', '')}\n"
            if self._use_wandb and self._wandb:
                self._wandb.log(
                    {
                        "final/best_success_metric": best_run_results["fitness"],
                        "final/best_rewards_correlation": best_run_results.get("rewards_correlation"),
                        "final/best_candidate_id": best_run_results.get("candidate_id"),
                        "final/best_candidate_checkpoint": best_run_results.get("checkpoint_file"),
                        "final/gpt_reward_method": best_run_results.get("reward"),
                        "final/task_feedback": best_run_results.get("feedback", ""),
                    }
                )
        else:
            output += "- No successful training run\n"
            if self._use_wandb and self._wandb:
                self._wandb.log({"final/best_success_metric": None, "final/best_rewards_correlation": None})

        print("Final results:\n", output)
        with open(f"{self._log_dir}/revolve_full_final_result.txt", "w") as f:
            f.write(output)
        with open(f"{self._log_dir}/best_run.json", "w") as f:
            json.dump(best_run_results, f, indent=2, default=str)
        if self._use_wandb and self._wandb:
            self._wandb.finish()
