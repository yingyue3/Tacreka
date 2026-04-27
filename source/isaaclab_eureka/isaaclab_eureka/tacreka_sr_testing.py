# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0
#
# Tacreka_SR (testing) — locked-feature-set evolution.
#
# Iteration 0 (exploration): the LLM proposes K candidate reward feature sets.
# Each set is implemented as a single reward function and trained. The set whose
# reward function reaches the best success metric is LOCKED for the rest of the run.
#
# Iteration 1+ (refinement): the LLM no longer proposes new feature sets. Instead,
# it refines EACH feature in the locked set INDIVIDUALLY (one LLM call per feature
# per parallel run) using the per-component values + task feedback observed in the
# previous iteration's best run. The refined locked set is then implemented as a
# new reward function. K such refinements are run in parallel each iteration.

import datetime
import json
import os
import time
from typing import Literal

# We import this here to avoid GLIBCXX_3.4.30 errors in Isaac Sim 5.1.
from isaaclab.app import AppLauncher  # noqa: F401  (kept for the same import-order reason as in tacreka_sr_auto.py)

from isaaclab_eureka import EUREKA_ROOT_DIR
from isaaclab_eureka.learning_curve_utils import export_learning_curve_artifacts, resolve_checkpoint_path
from isaaclab_eureka.config import (
    TASKS_CFG,
    TEST_FEATURE_GEN_INITIAL_PROMPT,
    TEST_FEATURE_GEN_PROMPT,
    TEST_FEATURE_AS_ONE_REWARD_PROMPT,
    TEST_FEATURE_AS_ONE_REWARD_INITIAL_PROMPT,
    TEST_PER_COMPONENT_REFINEMENT_PROMPT,
    TEST_PER_COMPONENT_REFINEMENT_SYSTEM_PROMPT,
    TEST_LOCKED_FEATURE_REWARD_PROMPT,
)
from isaaclab_eureka.managers import EurekaTaskManager, LLMManagerTac, RecordManagerQuad
from isaaclab_eureka.utils import load_tensorboard_logs


class Tacreka_SR:
    """Locked-feature-set evolution: explore once, refine each component individually."""

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
        num_parallel_runs: int = 3,
        use_wandb: bool = True,
        wandb_project: str = "isaaclab-eureka",
        wandb_entity: str = None,
        wandb_name: str = None,
        human_feedback: bool = False,  # kept for signature compatibility; not used here
    ):
        """Initialize the Tacreka_SR class.

        Args:
            task: The task to train the agent on.
            device: The device to run the training on.
            env_seed: The seed to use for the environment.
            rl_library: The RL library to use for training.
            max_training_iterations: The maximum number of training iterations for the RL agent.
            feedback_subsampling: The subsampling of the metrics given as feedback to the LLM.
            temperature: The temperature to use for the GPT model.
            gpt_model: The GPT model to use.
            num_parallel_runs: The number of parallel reward functions per iteration.
            use_wandb: Whether to use Weights & Biases for logging.
            wandb_project: The wandb project name.
            wandb_entity: The wandb entity/team name.
            wandb_name: The wandb run name. If None, uses timestamp.
            human_feedback: Unused in this evolution scheme; kept only for caller compatibility.
        """
        self.multi_gpus = False
        self._human_feedback = human_feedback  # currently unused; selection is metric-based.

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
        # K: number of parallel reward functions trained per iteration.
        self._num_parallel_runs = num_parallel_runs

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._log_dir = os.path.join(EUREKA_ROOT_DIR, "logs", "tacreka_sr", task, timestamp)
        self._rl_runs_dir = os.path.join(self._log_dir, "rl_runs")
        os.makedirs(self._log_dir)

        print("[INFO]: Setting up the LLM Manager...")
        self._llm_manager = LLMManagerTac(
            gpt_model=gpt_model,
            temperature=temperature,
            system_prompt=TEST_FEATURE_AS_ONE_REWARD_INITIAL_PROMPT,
            feature_prompt=TEST_FEATURE_GEN_INITIAL_PROMPT,
        )

        print("[INFO]: Setting up the Task Manager...")
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

        print("[INFO]: Setting up the Record Manager...")
        self._record_manager = RecordManagerQuad(
            task=task,
            num_envs=1,
            device=device,
            max_frames=900,
            num_episodes=1,
        )

        # We import here because doing this before launching Kit causes GLIBCXX errors.
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
                        "num_parallel_runs": num_parallel_runs,
                        "task_description": task_description,
                        "success_metric_to_win": self._success_metric_to_win,
                        "success_metric_tolerance": self._success_metric_tolerance,
                        "evolution_scheme": "locked-feature-set + per-component refinement",
                    },
                    dir=self._log_dir,
                )
                print(f"[INFO]: Weights & Biases logging initialized. Project: {wandb_project}, Run: {run_name}")
            except ImportError:
                print("[WARNING]: wandb not installed. Install with 'pip install wandb' to enable wandb logging.")
                self._use_wandb = False
                self._wandb = None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, max_eureka_iterations: int):
        """Run the locked-feature-set evolution loop.

        Args:
            max_eureka_iterations: The maximum number of evolution iterations to run.
        """
        import numpy as np

        K = self._num_parallel_runs

        feature_gen_user_prompt = TEST_FEATURE_GEN_PROMPT.format(
            task_description=self._task_description,
            success_metric_to_win=self._success_metric_to_win,
            get_observations_method_as_string=self._task_manager.get_observations_method_as_string,
        )

        # Locked across iterations once iter 0 finishes.
        locked_feature_set = None

        # Carried between iterations: feedback from the previous iteration's best run.
        last_run_task_feedback = None
        last_run_per_component_feedback = None  # dict[feature_name -> str]
        last_run_reward_code = None

        # Best run across all iterations.
        best_run_results = {"success_metric": None}

        for it in range(max_eureka_iterations):
            print(f"\n{'#' * 20} Running Tacreka Iteration {it} {'#' * 20} \n")

            feature_sets_this_iter: list[list[dict]] = []
            raw_feature_outputs_this_iter: list[str] = []
            reward_strings_this_iter: list[str] = []
            raw_reward_outputs_this_iter: list[str] = []

            if it == 0:
                feature_sets_this_iter, raw_feature_outputs_this_iter = self._iter0_generate_feature_sets(
                    feature_gen_user_prompt, K
                )
                reward_strings_this_iter, raw_reward_outputs_this_iter = self._generate_rewards_for_initial_sets(
                    feature_sets_this_iter
                )
            else:
                (
                    feature_sets_this_iter,
                    raw_feature_outputs_this_iter,
                    reward_strings_this_iter,
                    raw_reward_outputs_this_iter,
                ) = self._refine_locked_set_and_generate_rewards(
                    locked_feature_set=locked_feature_set,
                    last_run_task_feedback=last_run_task_feedback,
                    last_run_per_component_feedback=last_run_per_component_feedback,
                    last_run_reward_code=last_run_reward_code,
                    K=K,
                )

            # ---------- Train all K reward functions for this iteration. ----------
            print("+" * 10 + " Training Started " + "+" * 10)
            results: list[dict] = []
            if self.multi_gpus:
                results = self._task_manager.train(reward_strings_this_iter)
            else:
                for rs in reward_strings_this_iter:
                    results += self._task_manager.train([rs])
            time.sleep(1.0)  # let TensorBoard flush
            print("+" * 10 + " Training Ended, Evaluating " + "+" * 10)

            # ---------- Evaluate each run, attach metadata, find iter-best. ----------
            iter_best_idx, iter_best_metric = self._evaluate_and_attach_results(
                results=results,
                feature_sets_this_iter=feature_sets_this_iter,
                raw_feature_outputs_this_iter=raw_feature_outputs_this_iter,
                reward_strings_this_iter=reward_strings_this_iter,
                raw_reward_outputs_this_iter=raw_reward_outputs_this_iter,
            )

            self._log_iteration_results(it, results)

            # ---------- Lock feature set after iter 0. ----------
            if it == 0:
                if iter_best_idx is None:
                    print("[WARN] Iter 0: no successful run; locking the first generated set as fallback.")
                    iter_best_idx = 0
                locked_feature_set = feature_sets_this_iter[iter_best_idx]
                print(
                    f"[INFO] Locked feature set from iter 0 (run {iter_best_idx}, "
                    f"success_metric={iter_best_metric}):"
                )
                for f in locked_feature_set:
                    print(f"    - {f.get('feature_name', '<missing>')}")
                self._save_locked_feature_set(locked_feature_set, iter_best_idx, iter_best_metric)

            # ---------- Update best-overall + carry forward this iter's best feedback. ----------
            if iter_best_idx is not None and iter_best_metric is not None:
                self._update_best_overall(best_run_results, results[iter_best_idx], iter_best_metric, it, iter_best_idx)
                last_run_task_feedback = results[iter_best_idx]["eureka_task_feedback"]
                last_run_per_component_feedback = results[iter_best_idx]["per_component_feedback"]
                last_run_reward_code = results[iter_best_idx]["reward_code"]
            else:
                print(f"[WARN] Iter {it}: no successful run; reusing prior feedback for next iteration.")

            # ---------- Early stop. ----------
            if (
                best_run_results["success_metric"] is not None
                and np.abs(best_run_results["success_metric"] - self._success_metric_to_win)
                < self._success_metric_tolerance
            ):
                print(f"Task solved with success metric: {best_run_results['success_metric']}")
                break

        self._log_final_results(best_run_results)
        self._task_manager.close()
        self._record_manager.close()

    # ------------------------------------------------------------------
    # Per-iteration helpers
    # ------------------------------------------------------------------

    def _iter0_generate_feature_sets(self, feature_gen_user_prompt: str, K: int):
        """Iter 0: ask the LLM for K independent candidate reward feature sets."""
        print(f"[INFO] Iter 0: generating {K} candidate feature sets...")
        feature_gen_outputs = self._llm_manager.feature_gen(
            user_prompt=feature_gen_user_prompt,
            assistant_prompt=None,
            num_suggestion=K,
        )
        feature_sets = list(feature_gen_outputs["feature_strings"])
        raw_outputs = list(feature_gen_outputs["raw_outputs"])

        # Each entry in feature_sets is expected to be a list of feature dicts (the "features" array).
        # If the LLM returned a raw string due to JSON parse failure, fall back to an empty list.
        normalized_sets: list[list[dict]] = []
        for fs in feature_sets:
            if isinstance(fs, list) and all(isinstance(x, dict) for x in fs):
                normalized_sets.append(fs)
            else:
                print(f"[WARN] Iter 0: a generated feature set was not a valid list of dicts; got: {type(fs)}")
                normalized_sets.append([])
        return normalized_sets, raw_outputs

    def _generate_rewards_for_initial_sets(self, feature_sets: list[list[dict]]):
        """Iter 0: produce one reward function per candidate feature set."""
        reward_strings: list[str] = []
        raw_outputs: list[str] = []
        for k, feature_set in enumerate(feature_sets):
            self._llm_manager.single_feature_reset()
            rw_user_prompt = TEST_FEATURE_AS_ONE_REWARD_PROMPT.format(
                task_description=self._task_description,
                success_metric_to_win=self._success_metric_to_win,
                get_observations_method_as_string=self._task_manager.get_observations_method_as_string,
                FEATURES_JSON=json.dumps(feature_set, indent=2, default=str),
            )
            rw_output = self._llm_manager.single_feature_prompt(
                user_prompt=rw_user_prompt,
                assistant_prompt=None,
                num_suggestion=1,
            )
            reward_strings.append(rw_output["reward_strings"][0])
            raw_outputs.append(rw_output["raw_outputs"][0])
            print(f"[INFO] Iter 0: generated reward function for candidate set {k}.")
        return reward_strings, raw_outputs

    def _refine_locked_set_and_generate_rewards(
        self,
        locked_feature_set: list[dict],
        last_run_task_feedback: str,
        last_run_per_component_feedback: dict,
        last_run_reward_code: str,
        K: int,
    ):
        """Iter 1+: produce K refined feature sets (each via per-component LLM calls) + K reward functions."""
        print(
            f"[INFO] Refining each of the {len(locked_feature_set)} locked components individually "
            f"for each of {K} parallel runs..."
        )
        feature_sets: list[list[dict]] = []
        raw_feature_outputs: list[str] = []
        reward_strings: list[str] = []
        raw_reward_outputs: list[str] = []

        locked_feature_names = ", ".join(f.get("feature_name", "<unknown>") for f in locked_feature_set)
        last_task_fb = last_run_task_feedback or "(no task feedback available from a prior run)"
        last_reward_code = last_run_reward_code or "# (no previous reward code)"

        for k in range(K):
            refined_features: list[dict] = []
            for feat in locked_feature_set:
                feat_name = feat.get("feature_name", "<unknown>")
                component_fb = "(no component-level data captured for this feature)"
                if last_run_per_component_feedback is not None:
                    component_fb = last_run_per_component_feedback.get(feat_name, component_fb)

                refine_user_prompt = TEST_PER_COMPONENT_REFINEMENT_PROMPT.format(
                    task_description=self._task_description,
                    success_metric_to_win=self._success_metric_to_win,
                    get_observations_method_as_string=self._task_manager.get_observations_method_as_string,
                    feature_json=json.dumps(feat, indent=2, default=str),
                    locked_feature_names=locked_feature_names,
                    component_feedback=component_fb,
                    task_feedback=last_task_fb,
                )

                refine_out = self._llm_manager.refine_single_feature(
                    user_prompt=refine_user_prompt,
                    system_prompt=TEST_PER_COMPONENT_REFINEMENT_SYSTEM_PROMPT,
                    num_suggestion=1,
                )
                refined = refine_out["refined_features"][0] if refine_out["refined_features"] else None

                if (
                    refined is None
                    or not isinstance(refined, dict)
                    or refined.get("feature_name") != feat_name
                ):
                    # Contract violation: the LLM tried to rename or returned invalid JSON.
                    # Keep the original feature so the locked structure is preserved.
                    print(
                        f"[WARN] Refinement for feature '{feat_name}' violated the locked-name contract; "
                        f"keeping the previous version unchanged."
                    )
                    refined_features.append(feat)
                else:
                    refined_features.append(refined)

            feature_sets.append(refined_features)
            raw_feature_outputs.append(json.dumps({"features": refined_features}, default=str))

            # Generate a reward function from the refined locked set.
            self._llm_manager.single_feature_reset()
            rw_user_prompt = TEST_LOCKED_FEATURE_REWARD_PROMPT.format(
                task_description=self._task_description,
                get_observations_method_as_string=self._task_manager.get_observations_method_as_string,
                FEATURES_JSON=json.dumps(refined_features, indent=2, default=str),
                PREVIOUS_REWARD_CODE=last_reward_code,
                task_feedback=last_task_fb,
            )
            rw_output = self._llm_manager.single_feature_prompt(
                user_prompt=rw_user_prompt,
                assistant_prompt=None,
                num_suggestion=1,
            )
            reward_strings.append(rw_output["reward_strings"][0])
            raw_reward_outputs.append(rw_output["raw_outputs"][0])
            print(f"[INFO] Refined locked set {k}: produced reward function.")

        return feature_sets, raw_feature_outputs, reward_strings, raw_reward_outputs

    def _evaluate_and_attach_results(
        self,
        results: list[dict],
        feature_sets_this_iter: list[list[dict]],
        raw_feature_outputs_this_iter: list[str],
        reward_strings_this_iter: list[str],
        raw_reward_outputs_this_iter: list[str],
    ):
        """Score each parallel run; pick the iter-best. Attaches metadata to each result dict."""
        import numpy as np

        iter_best_idx = None
        iter_best_metric = None

        for idx, result in enumerate(results):
            results[idx]["feature_set"] = feature_sets_this_iter[idx]
            results[idx]["raw_feature_output"] = raw_feature_outputs_this_iter[idx]
            results[idx]["reward_code"] = reward_strings_this_iter[idx]
            results[idx]["raw_reward_output"] = raw_reward_outputs_this_iter[idx]

            if not result["success"]:
                results[idx]["eureka_task_feedback"] = (
                    f"Training failed: {result.get('exception', 'unknown error')}"
                )
                results[idx]["success_metric_max"] = None
                results[idx]["per_component_feedback"] = {}
                results[idx]["rewards_correlation"] = 0.0
                continue

            feedback_str, success_max, corr, _oracle = self._get_eureka_task_feedback(
                result["log_dir"], self._feedback_subsampling
            )
            per_comp = self._get_per_component_feedback(
                log_dir=result["log_dir"],
                feature_set=feature_sets_this_iter[idx],
                feedback_subsampling=self._feedback_subsampling,
            )
            results[idx]["eureka_task_feedback"] = feedback_str
            results[idx]["success_metric_max"] = success_max
            results[idx]["per_component_feedback"] = per_comp
            results[idx]["rewards_correlation"] = corr

            if success_max is not None and (
                iter_best_idx is None
                or np.abs(success_max - self._success_metric_to_win)
                < np.abs(iter_best_metric - self._success_metric_to_win)
            ):
                iter_best_idx = idx
                iter_best_metric = success_max

        return iter_best_idx, iter_best_metric

    def _update_best_overall(
        self,
        best_run_results: dict,
        best_run: dict,
        iter_best_metric: float,
        it: int,
        iter_best_idx: int,
    ):
        """Update the cross-iteration best-run dict if this iter beat it (closer to target)."""
        import numpy as np

        if best_run_results["success_metric"] is None or (
            np.abs(iter_best_metric - self._success_metric_to_win)
            < np.abs(best_run_results["success_metric"] - self._success_metric_to_win)
        ):
            best_run_results["success_metric"] = iter_best_metric
            best_run_results["task_feedback"] = best_run["eureka_task_feedback"]
            best_run_results["rewards_correlation"] = best_run["rewards_correlation"]
            best_run_results["feature_components"] = best_run["feature_set"]
            best_run_results["gpt_reward_method"] = best_run["reward_code"]
            best_run_results["training_log_dir"] = best_run.get("log_dir")
            best_run_results["training_run_dir"] = best_run.get("run_dir", best_run.get("log_dir"))
            best_run_results["checkpoint_file"] = best_run.get(
                "checkpoint_file"
            ) or resolve_checkpoint_path(best_run.get("run_dir", best_run.get("log_dir")))
            best_run_results["learning_curve"] = best_run.get("learning_curve")
            best_run_results["iteration"] = it
            best_run_results["run_idx"] = iter_best_idx
            print(f"[INFO] New best overall at iter {it} run {iter_best_idx}: {iter_best_metric}")

            if self._use_wandb and self._wandb:
                self._wandb.log(
                    {
                        "best/overall_success_metric": iter_best_metric,
                        "best/iteration": it,
                        "best/run_idx": iter_best_idx,
                    },
                    step=it,
                )

    def _save_locked_feature_set(self, locked_feature_set: list[dict], best_idx: int, best_metric):
        """Persist the locked feature set chosen at iter 0 for reproducibility/inspection."""
        path = os.path.join(self._log_dir, "locked_feature_set.json")
        with open(path, "w") as f:
            json.dump(
                {
                    "locked_at_iteration": 0,
                    "selected_run_idx": best_idx,
                    "success_metric_max": best_metric,
                    "features": locked_feature_set,
                },
                f,
                indent=2,
                default=str,
            )
        print(f"[INFO] Locked feature set saved to {path}")

    # ------------------------------------------------------------------
    # Tensorboard-derived feedback
    # ------------------------------------------------------------------

    def _get_eureka_task_feedback(self, log_dir: str, feedback_subsampling: int):
        """Build the overall task feedback string + headline metrics for a single training run."""
        import numpy as np

        data = load_tensorboard_logs(log_dir)

        eureka_rewards_data = next((data[key] for key in data if key.endswith("Eureka/eureka_total_rewards")), None)
        oracle_rewards_data = next((data[key] for key in data if key.endswith("Eureka/oracle_total_rewards")), None)

        if eureka_rewards_data is None or oracle_rewards_data is None:
            print(f"[WARNING] Missing reward data in TensorBoard logs. Available keys: {list(data.keys())}")
            rewards_correlation = 0.0
            oracle_last = None
        else:
            eureka_rewards = np.array(eureka_rewards_data)
            oracle_rewards = np.array(oracle_rewards_data)
            if (
                eureka_rewards.ndim == 0
                or oracle_rewards.ndim == 0
                or len(eureka_rewards) == 0
                or len(oracle_rewards) == 0
            ):
                rewards_correlation = 0.0
                oracle_last = None
            else:
                min_length = min(len(eureka_rewards), len(oracle_rewards))
                rewards_correlation = np.corrcoef(eureka_rewards[:min_length], oracle_rewards[:min_length])[0, 1]
                oracle_last = float(oracle_rewards[-1])

        success_metric_max = None
        total_feed_back_string = ""
        for metric_name, metric_data in data.items():
            if "Eureka/" not in metric_name:
                continue
            metric_data = metric_data[2:]  # drop the first two as outliers
            if not metric_data:
                continue
            short_name = metric_name.split("Eureka/", 1)[-1]
            metric_min = min(metric_data)
            metric_max = max(metric_data)
            metric_mean = sum(metric_data) / len(metric_data)
            metric_best = metric_data[np.abs(np.array(metric_data) - self._success_metric_to_win).argmin()]
            if short_name == "success_metric":
                short_name = "task_score"
                success_metric_max = metric_best
            data_string = [f"{x:.2f}" for x in metric_data[::feedback_subsampling]]
            line = (
                f"{short_name}: {data_string}, Min: {metric_min:.2f}, Max: {metric_max:.2f}, "
                f"Mean: {metric_mean:.2f}\n"
            )
            # If success metric is available we elide the (less informative) oracle line.
            if "Eureka/success_metric" in data and short_name == "oracle_total_rewards":
                line = ""
            total_feed_back_string += line

        total_feed_back_string += f"\nThe desired task_score to win is: {self._success_metric_to_win:.2f}\n"
        return total_feed_back_string, success_metric_max, rewards_correlation, oracle_last

    def _get_per_component_feedback(
        self,
        log_dir: str,
        feature_set: list[dict],
        feedback_subsampling: int,
    ) -> dict:
        """Return a dict feature_name -> per-component feedback string from the tensorboard logs.

        We try a few candidate tensorboard tag spellings per feature_name, since the LLM-generated
        reward dict keys do not always strictly follow the `r_<feature_name>` convention.
        """
        data = load_tensorboard_logs(log_dir)
        per_component: dict[str, str] = {}

        for feat in feature_set:
            feature_name = feat.get("feature_name")
            if not feature_name:
                continue

            candidates = [
                f"Eureka/{feature_name}",
                f"Eureka/r_{feature_name}",
                f"Eureka/reward_{feature_name}",
            ]
            matched_key = None
            matched_data = None
            for ck in candidates:
                if ck in data and len(data[ck]) > 2:
                    matched_key = ck
                    matched_data = data[ck]
                    break

            if matched_data is None:
                # Fallback: case-insensitive substring match against any Eureka/ tag.
                for k, v in data.items():
                    if "Eureka/" in k and feature_name.lower() in k.lower() and len(v) > 2:
                        matched_key = k
                        matched_data = v
                        break

            if matched_data is None:
                per_component[feature_name] = (
                    f"(no matching tensorboard component found for feature '{feature_name}')"
                )
                continue

            metric_data = matched_data[2:]
            m_min = min(metric_data)
            m_max = max(metric_data)
            m_mean = sum(metric_data) / len(metric_data)
            sub = [f"{x:.2f}" for x in metric_data[::feedback_subsampling]]
            tag_short = matched_key.split("Eureka/", 1)[-1]
            per_component[feature_name] = (
                f"{tag_short}: {sub}, Min: {m_min:.2f}, Max: {m_max:.2f}, Mean: {m_mean:.2f}"
            )

        return per_component

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_iteration_results(self, it: int, results: list[dict]):
        """Log the results of one evolution iteration to stdout, file, tensorboard, and wandb."""
        for idx, result in enumerate(results):
            print(f"{'*' * 20} Iteration {it} / Run {idx} {'*' * 20}")
            if result["success"]:
                print(f"Training succeeded. Metrics:\n{result['eureka_task_feedback']}")
                print(f"Reward correlation with oracle: {result['rewards_correlation']}")
            else:
                print(f"Training failed: {result.get('exception', 'unknown')}")

        with open(f"{self._log_dir}/eureka_iterations.txt", "a") as f:
            for idx, result in enumerate(results):
                f.write(f"{'#' * 20} Iteration {it} / Run {idx} {'#' * 20}\n\n")
                feature_set_str = json.dumps(result.get("feature_set", []), indent=2, default=str)
                f.write(f"- Feature set:\n{feature_set_str}\n")
                f.write(f"- Raw LLM reward output:\n{result.get('raw_reward_output', '')}\n")
                if result["success"]:
                    f.write(f"- Training feedback:\n{result['eureka_task_feedback']}\n")
                    f.write(f"- Reward correlation: {result['rewards_correlation']}\n")
                    f.write("- Per-component feedback:\n")
                    for fname, fb in result.get("per_component_feedback", {}).items():
                        f.write(f"    [{fname}] {fb}\n")
                    success_metric_value = result.get("success_metric_max")
                    if success_metric_value is None:
                        success_metric_value = 0.0
                    self._tensorboard_writer.add_scalar(f"Run_{idx}/success_metric", success_metric_value, it)
                    if self._use_wandb and self._wandb:
                        self._wandb.log(
                            {
                                f"Run_{idx}/success_metric": success_metric_value,
                                f"Run_{idx}/rewards_correlation": result.get("rewards_correlation", 0.0),
                            },
                            step=it,
                        )
                else:
                    f.write(f"- Training failed: {result.get('exception', 'unknown')}\n")
                    self._tensorboard_writer.add_scalar(f"Run_{idx}/success_metric", 0.0, it)
                    if self._use_wandb and self._wandb:
                        self._wandb.log({f"Run_{idx}/success_metric": 0.0}, step=it)

                self._tensorboard_writer.add_text(
                    f"Run_{idx}/raw_llm_reward_output", result.get("raw_reward_output", ""), it
                )
                self._tensorboard_writer.add_text(
                    f"Run_{idx}/raw_feature_set", result.get("raw_feature_output", ""), it
                )
                f.write("\n")

    def _log_final_results(self, best_run_results: dict):
        """Log the final results of the evolution run."""
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
            output += f"- GPT reward method:\n{best_run_results['gpt_reward_method']}\n"
            output += f"- Locked + refined feature components: {json.dumps(best_run_results['feature_components'], indent=2, default=str)}\n"
            output += f"- Best training log dir: {best_run_results.get('training_log_dir', 'unknown')}\n"
            output += f"- Best training run dir: {best_run_results.get('training_run_dir', 'unknown')}\n"
            output += f"- Best checkpoint: {best_run_results.get('checkpoint_file', 'unknown')}\n"
            learning_curve_path = (best_run_results.get("best_learning_curve") or {}).get("plot_path", "unknown")
            output += f"- Best learning curve plot: {learning_curve_path}\n"
            output += f"- Task metrics:\n{best_run_results['task_feedback']}\n"

            if self._use_wandb and self._wandb:
                self._wandb.log(
                    {
                        "final/best_success_metric": best_run_results["success_metric"],
                        "final/gpt_reward_method": best_run_results["gpt_reward_method"],
                        "final/task_feedback": best_run_results["task_feedback"],
                    }
                )
        else:
            output += "- No successful training run\n"
            if self._use_wandb and self._wandb:
                self._wandb.log({"final/best_success_metric": None})

        print("Final results:\n", output)

        with open(f"{self._log_dir}/eureka_final_result.txt", "w") as f:
            f.write(output)
        with open(f"{self._log_dir}/best_run.json", "w") as f:
            json.dump(best_run_results, f, indent=2, default=str)

        if self._use_wandb and self._wandb:
            self._wandb.finish()
