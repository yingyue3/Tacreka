import os
import random
import sys
from typing import List, Tuple, Dict

import numpy as np
from absl import logging

from isaaclab_eureka.revolve_full.entities import Island


def normalized(x: List[float], temp: float = 1):
    """Numerically stable probability normalization with safe fallbacks."""
    values = np.asarray(x, dtype=np.float64)
    if values.size == 0:
        return values

    if not np.isfinite(temp) or temp <= 0:
        temp = 1.0

    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return np.full(values.shape, 1.0 / values.size, dtype=np.float64)

    finite_min = np.min(values[finite_mask])
    values = np.where(finite_mask, values, finite_min)

    # Stable softmax: subtract max to avoid overflow/underflow issues.
    shifted = (values - np.max(values)) / temp
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        weights = np.exp(shifted)
    weights = np.where(np.isfinite(weights), weights, 0.0)

    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        return np.full(values.shape, 1.0 / values.size, dtype=np.float64)

    probs = weights / total
    probs = np.where(np.isfinite(probs), probs, 0.0)
    prob_sum = float(np.sum(probs))
    if not np.isfinite(prob_sum) or prob_sum <= 0.0:
        return np.full(values.shape, 1.0 / values.size, dtype=np.float64)
    return probs / prob_sum


class RevolveDatabase:
    """
    Simplified REvolve-style population manager for island-based sampling.
    """

    def __init__(
        self,
        num_islands: int,
        max_size: int,
        crossover_prob: float,
        migration_prob: float,
        load_islands: bool,
        reward_fn_dir: str,
    ):
        self.reward_fn_dir = reward_fn_dir
        self.num_islands = num_islands
        self.max_size = max_size
        self.crossover_prob = crossover_prob
        self.migration_prob = migration_prob
        self.heuristic_dir = reward_fn_dir

        self._islands: List[Island] = []
        if load_islands:
            for island_id in range(self.num_islands):
                loaded_island = Island.load_island(self.reward_fn_dir, island_id)
                self._islands.append(loaded_island)
        else:
            self._islands = [
                Island(island_id, [], [], [], [], [], self.heuristic_dir)
                for island_id in range(self.num_islands)
            ]

    def seed_islands(
        self,
        generation_ids: List[int],
        counter_ids: List[int],
        rew_fn_strings: List[str],
        fitness_scores: List[float],
        metrics_dicts: List[dict],
        island_ids: List[int],
    ):
        for (
            generation_id,
            counter_id,
            rew_fn_string,
            fitness_score,
            metrics_dict,
            island_id,
        ) in zip(
            generation_ids,
            counter_ids,
            rew_fn_strings,
            fitness_scores,
            metrics_dicts,
            island_ids,
        ):
            logging.info(
                f"Inside seed_islands: island_id={island_id}, generation_id={generation_id}, counter_id={counter_id}"
            )
            self._islands[island_id].register_individual_in_island(
                generation_id, counter_id, rew_fn_string, fitness_score, metrics_dict
            )

    def add_individuals_to_islands(
        self,
        generation_ids: List[int],
        counter_ids: List[int],
        rew_fn_strings: List[str],
        fitness_scores: List[float],
        metrics_dicts: List[dict],
        island_ids: List[int],
    ):
        for (
            generation_id,
            counter_id,
            rew_fn_string,
            fitness_score,
            island_id,
            metrics_dict,
        ) in zip(
            generation_ids,
            counter_ids,
            rew_fn_strings,
            fitness_scores,
            island_ids,
            metrics_dicts,
        ):
            if self._islands[island_id].size != 0:
                island_avg_fitness_score = self._islands[
                    island_id
                ].average_fitness_score
            else:
                island_avg_fitness_score = -sys.maxsize - 1

            if fitness_score >= island_avg_fitness_score:
                self._islands[island_id].register_individual_in_island(
                    generation_id,
                    counter_id,
                    rew_fn_string,
                    fitness_score,
                    metrics_dict,
                )
                logging.info(
                    "Average score of island %d increased to %s",
                    island_id,
                    self._islands[island_id].average_fitness_score,
                )
            else:
                logging.info(
                    "Fitness score %s for individual lower than average "
                    "Island %d fitness %s, discarding",
                    fitness_score,
                    island_id,
                    island_avg_fitness_score,
                )
                reward_history_path = (
                    f"{self.reward_fn_dir}/island_{island_id}/reward_history/"
                    f"{generation_id}_{counter_id}.json"
                )
                model_checkpoint_path = (
                    f"{self.reward_fn_dir}/island_{island_id}/model_checkpoints/"
                    f"{generation_id}_{counter_id}.h5"
                )
                RevolveDatabase.delete_file(
                    reward_history_path, "reward history (.json) file"
                )
                RevolveDatabase.delete_file(
                    model_checkpoint_path, "model checkpoint (.h5) file"
                )

            if self._islands[island_id].size > self.max_size:
                logging.info(
                    "Exceeded maximum size on island %d, "
                    "discarding individual with lowest score",
                    island_id,
                )
                while self._islands[island_id].size > self.max_size:
                    self._islands[island_id].remove_lowest()

        if random.random() <= self.migration_prob and len(self._islands) > 1:
            self.reset_islands()

    def reset_islands(self):
        print("============ Resetting Island ============")
        indices_sorted_by_score = np.argsort(
            np.array([island.best_fitness_score for island in self._islands])
            + np.random.randn(len(self._islands)) * 1e-6
        )
        num_islands_to_reset = len(self._islands) // 2
        reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
        keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]
        for reset_island_id in reset_islands_ids:
            self._islands[reset_island_id].only_keep_best()
            founder_island_id = np.random.choice(keep_islands_ids)
            founder_island = self._islands[founder_island_id]
            repeats = 0
            while founder_island.size <= 1:
                founder_island_id = np.random.choice(keep_islands_ids)
                founder_island = self._islands[founder_island_id]
                repeats += 1
                if repeats >= 10:
                    break
            if repeats >= 10:
                continue
            founder_individual = founder_island.fittest_individual
            while founder_individual == founder_island.fittest_individual:
                founder_individual = random.choices(
                    founder_island.individuals,
                    normalized(founder_island.fitness_scores),
                )[0]
            logging.info(
                f"Migrating individual from Island {founder_island_id} to Island {reset_island_id}"
            )
            self._islands[reset_island_id].migrate_fn(founder_individual)
            self._islands[founder_island_id].remove_individual(founder_individual)

    def sample_in_context(
        self, num_samples: Dict, temperature: float
    ) -> Tuple[List[Tuple[str, float]], int, str]:
        island_avg_scores = [
            self._islands[island_id].average_fitness_score
            for island_id in range(self.num_islands)
        ]
        average_fitness_scores = normalized(island_avg_scores, temperature)

        operator = "mutation" if random.random() >= self.crossover_prob else "crossover"
        num_in_context_samples = (
            num_samples["mutation"]
            if operator == "mutation"
            else num_samples["crossover"]
        )

        # Prefer islands that can satisfy requested sample count.
        eligible_islands = [
            (idx, island) for idx, island in enumerate(self._islands) if island.size >= num_in_context_samples
        ]
        if not eligible_islands:
            # Relax to any non-empty island, and downsize requested context count if needed.
            eligible_islands = [(idx, island) for idx, island in enumerate(self._islands) if island.size > 0]
            if not eligible_islands:
                logging.warning(
                    "No non-empty islands available for in-context sampling; returning no examples from island 0."
                )
                return [], 0, "mutation"
            num_in_context_samples = min(num_in_context_samples, max(island.size for _, island in eligible_islands))

        eligible_ids = [idx for idx, _ in eligible_islands]
        eligible_weights = normalized([island_avg_scores[idx] for idx in eligible_ids], temperature)
        sampled_eligible_idx = random.choices(range(len(eligible_islands)), weights=eligible_weights)[0]
        sampled_island_id, sampled_island = eligible_islands[sampled_eligible_idx]

        if sampled_island.size == 0 or num_in_context_samples <= 0:
            logging.warning(
                "Sampled island %d has no individuals for in-context sampling; returning empty sample list.",
                sampled_island_id,
            )
            return [], sampled_island_id, operator

        num_in_context_samples = min(num_in_context_samples, sampled_island.size)
        in_context_sample_ids = np.random.choice(
            range(sampled_island.size),
            p=normalized(sampled_island.fitness_scores, temperature),
            size=num_in_context_samples,
            replace=False,
        )
        in_context_samples = list(
            zip(
                np.array(sampled_island.fn_file_paths)[in_context_sample_ids],
                np.array(sampled_island.fitness_scores)[in_context_sample_ids],
            )
        )
        logging.info(f"{operator.capitalize()} | sampled island: {sampled_island_id}")
        return in_context_samples, sampled_island_id, operator

    @staticmethod
    def delete_file(filepath: str, filetype: str):
        if os.path.exists(filepath):
            logging.info(f"Removing {filetype} from {filepath}.")
            os.remove(filepath)
        else:
            logging.info(f"{filetype} does not exist in {filepath}.")
