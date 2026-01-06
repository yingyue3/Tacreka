# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

import datetime
import math
import os
import random
from functools import partial
from typing import Dict, List, Optional

from isaaclab_eureka import EUREKA_ROOT_DIR
from isaaclab_eureka.config import (
    DIRECT_WORKFLOW_INITIAL_PROMPT,
    DIRECT_WORKFLOW_TASK_PROMPT,
    TASKS_CFG,
)
from isaaclab_eureka.managers import EurekaTaskManager, LLMManager
from isaaclab_eureka.revolve_full.database import RevolveDatabase
from isaaclab_eureka.utils import load_tensorboard_logs


def _linear_decay(iteration: int, initial: float, final: float, num_iterations: int):
    return initial - (initial - final) * iteration / max(num_iterations, 1)


def _append_in_context_prompt(base_prompt: str, samples: List[tuple[str, float]]) -> str:
    if not samples:
        return base_prompt
    context_lines = ["\nPrevious reward functions and scores:"]
    for filename, fitness_score in samples:
        try:
            with open(filename, "r") as f:
                fn_str = f.read()
            context_lines.append(f"\nFitness: {fitness_score:.3f}\n```python\n{fn_str}\n```")
        except FileNotFoundError:
            continue
    return base_prompt + "\n" + "\n".join(context_lines)


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
        few_shot: Optional[Dict[str, int]] = None,
        temperature_final: float = 1.0,
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
        self._temperature_initial = temperature
        self._temperature_final = temperature_final
        self._num_islands = num_islands
        self._max_island_size = max_island_size
        self._crossover_prob = crossover_prob
        self._migration_prob = migration_prob

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._log_dir = os.path.join(EUREKA_ROOT_DIR, "logs", "revolve_full", task, timestamp)
        self._db_dir = os.path.join(self._log_dir, "database")
        os.makedirs(self._db_dir, exist_ok=True)

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
                    },
                    dir=self._log_dir,
                )
            except ImportError:
                print("[WARNING]: wandb not installed. Install with 'pip install wandb' to enable wandb logging.")
                self._use_wandb = False
                self._wandb = None

    def run(self):
        best_overall = {"fitness": None, "reward": None, "feedback": ""}
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

            for counter_id in range(self._individuals_per_generation):
                if generation_id == 0:
                    island_id = random.choice(range(database.num_islands))
                    in_context_samples = []
                else:
                    in_context_samples, island_id, _ = database.sample_in_context(
                        self._few_shot, temperature
                    )
                island_ids.append(island_id)
                prompt_with_context = _append_in_context_prompt(base_user_prompt, in_context_samples)

                llm_manager = self._llm_factory(temperature=temperature)
                llm_outputs = llm_manager.prompt(user_prompt=prompt_with_context)
                reward_string = llm_outputs["reward_strings"][0]
                if reward_string is None or reward_string.strip() == "":
                    print(f"[WARN] Empty reward string for generation {generation_id}, counter {counter_id}. Skipping.")
                    continue

                result = self._task_manager.train([reward_string])[0]
                fitness = 0.0
                feedback = ""
                correlation = 0.0
                if result["success"]:
                    feedback, success_metric_max, correlation = self._get_eureka_task_feedback(
                        result["log_dir"], feedback_subsampling=10
                    )
                    fitness = success_metric_max if success_metric_max is not None else 0.0
                else:
                    feedback = result.get("exception", "")
                    success_metric_max = None

                metrics_dict = {
                    "fitness": fitness,
                    "rewards_correlation": correlation,
                    "success": result["success"],
                }
                metrics_dicts.append(metrics_dict)
                rew_fn_strings.append(reward_string)
                fitness_scores.append(fitness)
                counter_ids.append(counter_id)

                if best_overall["fitness"] is None or (
                    success_metric_max is not None
                    and abs(success_metric_max - self._task_cfg["success_metric_to_win"])
                    < abs(best_overall["fitness"] - self._task_cfg["success_metric_to_win"])
                ):
                    best_overall["fitness"] = success_metric_max
                    best_overall["reward"] = reward_string
                    best_overall["feedback"] = feedback

            if len(rew_fn_strings) == 0:
                print("[WARN] No valid reward functions generated; skipping generation update.")
                continue

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

    def _get_eureka_task_feedback(self, log_dir: str, feedback_subsampling: int) -> tuple[str, float, float]:
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
                        abs(np.array(metric_data) - self._task_cfg["success_metric_to_win"]).argmin()
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
        total_feed_back_string += f"\nThe desired task_score to win is: {self._task_cfg['success_metric_to_win']:.2f}\n"
        return total_feed_back_string, success_metric_max, rewards_correlation

    def _log_final_results(self, best_run_results: Dict) -> None:
        output = ""
        if best_run_results.get("fitness") is not None:
            output += f"- Success metric: {best_run_results['fitness']}\n"
            output += f"- GPT reward method:\n{best_run_results.get('reward')}\n"
            output += f"- Task metrics:\n{best_run_results.get('feedback', '')}\n"
            if self._use_wandb and self._wandb:
                self._wandb.log(
                    {
                        "final/best_success_metric": best_run_results["fitness"],
                        "final/gpt_reward_method": best_run_results.get("reward"),
                        "final/task_feedback": best_run_results.get("feedback", ""),
                    }
                )
        else:
            output += "- No successful training run\n"
            if self._use_wandb and self._wandb:
                self._wandb.log({"final/best_success_metric": None})

        print("Final results:\n", output)
        with open(f"{self._log_dir}/revolve_full_final_result.txt", "w") as f:
            f.write(output)
        if self._use_wandb and self._wandb:
            self._wandb.finish()
