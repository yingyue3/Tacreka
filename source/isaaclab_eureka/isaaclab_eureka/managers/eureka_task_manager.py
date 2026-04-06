# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

import json
import math
import multiprocessing
import os
import subprocess
import sys
import tempfile
import traceback
import types
from contextlib import nullcontext
from datetime import datetime
from typing import Literal

from isaaclab_eureka.learning_curve_utils import export_learning_curve_artifacts, resolve_checkpoint_path
from isaaclab_eureka.utils import MuteOutput, bootstrap_observations_via_subprocess, resolve_sim_device

TEMPLATE_REWARD_STRING = """
from {module_name} import *
import torch

def _get_rewards(self):
    rewards_oracle = self._get_rewards_oracle()
    rewards_eureka, rewards_dict = self._get_rewards_eureka()
    self._eureka_episode_sums["eureka_total_rewards"] += rewards_eureka
    self._eureka_episode_sums["oracle_total_rewards"] += rewards_oracle
    for key in rewards_dict.keys():
        if key not in self._eureka_episode_sums:
            self._eureka_episode_sums[key] = torch.zeros(self.num_envs, device=self.device)
        self._eureka_episode_sums[key] += rewards_dict[key]
    return rewards_eureka
"""


TEMPLATE_RESET_STRING = """
from {module_name} import *

@torch.inference_mode()
def _reset_idx(self, env_ids):
    if env_ids is None or len(env_ids) == self.num_envs:
        env_ids = torch.arange(self.num_envs, device=self.device)
    extras = dict()
    {success_metric}
    self._reset_idx_original(env_ids)
    if not "log" in self.extras:
        self.extras["log"] = dict()
    for key in self._eureka_episode_sums.keys():
        episodic_sum_avg = torch.mean(self._eureka_episode_sums[key][env_ids])
        extras["Eureka/"+key] = episodic_sum_avg / self.max_episode_length_s
        self._eureka_episode_sums[key][env_ids] = 0.0
    self.extras["log"].update(extras)
"""


class EurekaTaskManager:
    """Manages Isaac Lab task training for LLM-generated reward functions."""

    def __init__(
        self,
        task: str,
        rl_library: Literal["rsl_rl", "rl_games"] = "rsl_rl",
        num_processes: int = 1,
        device: str = "cuda",
        env_seed: int = 42,
        max_training_iterations: int = 100,
        success_metric_string: str = "",
        log_namespace: str = "eureka",
        rl_log_root_dir: str | None = None,
        num_seeds_per_reward: int = 1,
    ):
        self._task = task
        self._rl_library = rl_library
        self._num_processes = num_processes
        self._device = device
        self._max_training_iterations = max_training_iterations
        self._success_metric_string = success_metric_string
        self._env_seed = env_seed
        self._rl_log_root_dir = os.path.abspath(rl_log_root_dir) if rl_log_root_dir else None
        sanitized_namespace = str(log_namespace).strip().replace(" ", "_")
        self._log_namespace = sanitized_namespace if sanitized_namespace else "eureka"
        self._num_seeds_per_reward = max(1, int(num_seeds_per_reward))
        self._current_env_seed = None
        if self._success_metric_string:
            self._success_metric_string = "extras['Eureka/success_metric'] = " + self._success_metric_string

        self._processes = {}
        self._rewards_queues = [multiprocessing.Queue() for _ in range(self._num_processes)]
        self._results_queue = multiprocessing.Queue()
        self.termination_event = multiprocessing.Event()
        self._get_observations_as_string = bootstrap_observations_via_subprocess(
            self._task, self._device, self._env_seed
        )

        for idx in range(self._num_processes):
            process = multiprocessing.Process(target=self._worker, args=(idx, self._rewards_queues[idx]))
            self._processes[idx] = process
            process.start()

    @property
    def get_observations_method_as_string(self) -> str:
        return self._get_observations_as_string

    def close(self):
        self.termination_event.set()
        for rewards_queue in self._rewards_queues:
            rewards_queue.put("Stop")
        for process in self._processes.values():
            process.join()

    def train(self, get_rewards_method_as_string: list[str]) -> list[dict]:
        if len(get_rewards_method_as_string) != self._num_processes:
            raise ValueError(
                f"Number of reward methods in the list ({len(get_rewards_method_as_string)}) does not match the number"
                f" of processes ({self._num_processes})."
            )

        for idx, rewards_queue in enumerate(self._rewards_queues):
            rewards_queue.put(get_rewards_method_as_string[idx])

        results = [None] * self._num_processes
        for _ in range(self._num_processes):
            idx, result = self._results_queue.get()
            results[idx] = result
        return results

    def _worker(self, idx: int, rewards_queue: multiprocessing.Queue):
        self._idx = idx

        while not self.termination_event.is_set():
            reward_func_string = rewards_queue.get()
            if reward_func_string == "Stop":
                break

            if isinstance(reward_func_string, str) and reward_func_string.startswith("def _get_rewards_eureka(self)"):
                try:
                    context = MuteOutput() if self._idx > 0 else nullcontext()
                    with context:
                        result = self._train_reward_across_seeds(reward_func_string)
                except Exception as error:
                    result = {"success": False, "exception": str(error)}
                    print(traceback.format_exc())
            else:
                result = {
                    "success": False,
                    "exception": (
                        "The reward function must be a string that starts with 'def _get_rewards_eureka(self)'."
                    ),
                }

            self._results_queue.put((self._idx, result))

        print(f"[INFO]: Run {self._idx} terminated.")
        self._close_environment()

    def _resolve_device(self, device: str) -> str:
        return resolve_sim_device(device)

    def _ensure_simulation_app(self, device: str):
        if hasattr(self, "_simulation_app"):
            return
        from isaaclab.app import AppLauncher

        self._resolved_device = self._resolve_device(device)
        app_launcher = AppLauncher(headless=True, device=self._resolved_device)
        self._simulation_app = app_launcher.app

    def _close_environment(self):
        env = getattr(self, "_env", None)
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
            delattr(self, "_env")
        if hasattr(self, "_simulation_app"):
            try:
                self._simulation_app.close()
            except Exception:
                pass
            delattr(self, "_simulation_app")
        self._current_env_seed = None

    def _create_environment(self, env_seed: int):
        if hasattr(self, "_env") and self._current_env_seed == env_seed:
            return

        self._close_environment()
        self._ensure_simulation_app(self._device)

        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401
        from isaaclab.envs import DirectRLEnvCfg
        from isaaclab_tasks.utils import parse_env_cfg

        env_cfg: DirectRLEnvCfg = parse_env_cfg(self._task)
        env_cfg.sim.device = self._resolved_device
        env_cfg.seed = env_seed
        self._env = gym.make(self._task, cfg=env_cfg)
        self._current_env_seed = env_seed

    def _prepare_eureka_environment(self, get_rewards_method_as_string: str):
        import torch

        env = self._env.unwrapped
        namespace = {}
        if not hasattr(env, "_get_rewards_eureka"):
            env._get_rewards_oracle = env._get_rewards
            env._reset_idx_original = env._reset_idx
            template_reward_string_with_module = TEMPLATE_REWARD_STRING.format(module_name=env.__module__)
            exec(template_reward_string_with_module, namespace)
            setattr(env, "_get_rewards", types.MethodType(namespace["_get_rewards"], env))
            template_reset_string_with_success_metric = TEMPLATE_RESET_STRING.format(
                module_name=env.__module__, success_metric=self._success_metric_string
            )
            if self._rl_library == "rl_games":
                template_reset_string_with_success_metric = template_reset_string_with_success_metric.replace(
                    "@torch.inference_mode()", ""
                )
            exec(template_reset_string_with_success_metric, namespace)
            setattr(env, "_reset_idx", types.MethodType(namespace["_reset_idx"], env))

        get_rewards_method_as_string = f"from {env.__module__} import * \nimport torch\n" + get_rewards_method_as_string
        exec(get_rewards_method_as_string, namespace)
        setattr(env, "_get_rewards_eureka", types.MethodType(namespace["_get_rewards_eureka"], env))

        env._eureka_episode_sums = {
            "eureka_total_rewards": torch.zeros(env.num_envs, device=env.device),
            "oracle_total_rewards": torch.zeros(env.num_envs, device=env.device),
        }

    def _train_reward_across_seeds(self, reward_func_string: str) -> dict:
        seed_results: list[dict] = []
        for seed_index in range(self._num_seeds_per_reward):
            seed = int(self._env_seed) + seed_index
            seed_results.append(self._run_seed_in_subprocess(reward_func_string, seed, seed_index))

        successful_seed_results = [result for result in seed_results if result.get("success")]
        if not successful_seed_results:
            failure_messages = [result.get("exception", "unknown seed failure") for result in seed_results]
            return {
                "success": False,
                "exception": " | ".join(failure_messages),
                "seed_results": seed_results,
                "num_reward_seeds": self._num_seeds_per_reward,
            }

        seed_log_dirs = [result["log_dir"] for result in successful_seed_results if result.get("log_dir")]
        seed_run_dirs = [result["run_dir"] for result in successful_seed_results if result.get("run_dir")]
        seed_checkpoint_files = [
            result["checkpoint_file"] for result in successful_seed_results if result.get("checkpoint_file")
        ]

        aggregate_result = {
            "success": True,
            "log_dir": seed_log_dirs[0],
            "run_dir": seed_run_dirs[0] if seed_run_dirs else seed_log_dirs[0],
            "seed_results": seed_results,
            "seed_log_dirs": seed_log_dirs,
            "seed_run_dirs": seed_run_dirs,
            "seed_checkpoint_files": seed_checkpoint_files,
            "num_reward_seeds": self._num_seeds_per_reward,
        }

        try:
            aggregate_output_dir = os.path.join(aggregate_result["run_dir"], "aggregate_learning_curves")
            aggregate_learning_curve = export_learning_curve_artifacts(
                seed_log_dirs,
                output_dir=aggregate_output_dir,
                run_name=os.path.basename(aggregate_result["run_dir"]),
            )
            if aggregate_learning_curve is not None:
                aggregate_result["learning_curve"] = aggregate_learning_curve
        except Exception as plot_error:
            aggregate_result["learning_curve_error"] = str(plot_error)

        return aggregate_result

    def _run_seed_in_subprocess(self, reward_func_string: str, seed: int, seed_index: int) -> dict:
        with tempfile.TemporaryDirectory(prefix=f"{self._log_namespace}_seed_{self._idx}_") as temp_dir:
            reward_file = os.path.join(temp_dir, "reward.py")
            result_file = os.path.join(temp_dir, "result.json")
            with open(reward_file, "w") as file:
                file.write(reward_func_string)

            cmd = [
                sys.executable,
                "-m",
                "isaaclab_eureka.seed_training_worker",
                "--manager-type",
                "eureka",
                "--task",
                self._task,
                "--rl-library",
                self._rl_library,
                "--device",
                self._device,
                "--seed",
                str(seed),
                "--seed-index",
                str(seed_index),
                "--run-index",
                str(self._idx),
                "--max-training-iterations",
                str(self._max_training_iterations),
                "--reward-file",
                reward_file,
                "--result-file",
                result_file,
                "--success-metric-string",
                self._success_metric_string,
                "--log-namespace",
                self._log_namespace,
            ]
            if self._rl_log_root_dir:
                cmd.extend(["--rl-log-root-dir", self._rl_log_root_dir])

            completed = subprocess.run(cmd)

            if os.path.exists(result_file):
                with open(result_file) as file:
                    seed_result = json.load(file)
            else:
                seed_result = {
                    "success": False,
                    "seed": seed,
                    "seed_index": seed_index,
                    "exception": f"Seed subprocess failed with return code {completed.returncode}.",
                }

            if completed.returncode != 0 and seed_result.get("success"):
                seed_result["subprocess_return_code"] = completed.returncode
            return seed_result

    def _set_training_seed(self, seed: int):
        import random

        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _run_training(self, seed: int, framework: Literal["rsl_rl", "rl_games"] = "rsl_rl"):
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        self._set_training_seed(seed)

        if self._rl_library == "rsl_rl":
            from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
            from rsl_rl.runners import OnPolicyRunner

            agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(self._task, "rsl_rl_cfg_entry_point")
            agent_cfg.device = getattr(self, "_resolved_device", self._resolve_device(self._device))
            agent_cfg.max_iterations = self._max_training_iterations
            if hasattr(agent_cfg, "seed"):
                agent_cfg.seed = seed

            if self._rl_log_root_dir:
                log_root_path = self._rl_log_root_dir
            else:
                log_root_path = os.path.join(
                    "logs", "rl_runs", f"rsl_rl_{self._log_namespace}", agent_cfg.experiment_name
                )
            log_root_path = os.path.abspath(log_root_path)
            os.makedirs(log_root_path, exist_ok=True)
            print(f"[INFO] Logging experiment in directory: {log_root_path}")

            log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_Run-{self._idx}_Seed-{seed}"
            if self._rl_log_root_dir:
                experiment_name = str(agent_cfg.experiment_name).replace(os.sep, "_").replace(" ", "_")
                log_dir = f"rsl_rl_{experiment_name}_{log_dir}"
            if agent_cfg.run_name:
                log_dir += f"_{agent_cfg.run_name}"
            self._run_dir = os.path.join(log_root_path, log_dir)
            self._log_dir = self._run_dir

            env = RslRlVecEnvWrapper(self._env)
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=self._log_dir, device=agent_cfg.device)
            runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

        elif self._rl_library == "rl_games":
            from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
            from rl_games.common import env_configurations, vecenv
            from rl_games.common.algo_observer import IsaacAlgoObserver
            from rl_games.torch_runner import Runner

            agent_cfg = load_cfg_from_registry(self._task, "rl_games_cfg_entry_point")
            agent_cfg["params"]["config"]["max_epochs"] = self._max_training_iterations
            agent_cfg["params"]["config"]["device"] = getattr(self, "_resolved_device", self._resolve_device(self._device))
            agent_cfg["params"]["config"]["device_name"] = agent_cfg["params"]["config"]["device"]
            agent_cfg["params"]["seed"] = seed
            agent_cfg["params"]["config"]["seed"] = seed

            if self._rl_log_root_dir:
                log_root_path = self._rl_log_root_dir
            else:
                log_root_path = os.path.join(
                    "logs", "rl_runs", f"rl_games_{self._log_namespace}", agent_cfg["params"]["config"]["name"]
                )
            log_root_path = os.path.abspath(log_root_path)
            os.makedirs(log_root_path, exist_ok=True)
            print(f"[INFO] Logging experiment in directory: {log_root_path}")

            log_dir = (
                agent_cfg["params"]["config"].get("full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
                + f"_Run-{self._idx}_Seed-{seed}"
            )
            if self._rl_log_root_dir:
                experiment_name = str(agent_cfg["params"]["config"]["name"]).replace(os.sep, "_").replace(" ", "_")
                log_dir = f"rl_games_{experiment_name}_{log_dir}"
            agent_cfg["params"]["config"]["train_dir"] = log_root_path
            agent_cfg["params"]["config"]["full_experiment_name"] = log_dir
            self._run_dir = os.path.join(log_root_path, log_dir)
            self._log_dir = os.path.join(self._run_dir, "summaries")
            clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
            clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
            env = RlGamesVecEnvWrapper(self._env, agent_cfg["params"]["config"]["device"], clip_obs, clip_actions)

            vecenv.register(
                "IsaacRlgWrapper",
                lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
            )
            env_configurations.register(
                "rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env}
            )

            agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
            runner = Runner(IsaacAlgoObserver())
            runner.load(agent_cfg)
            runner.reset()
            runner.run({"train": True, "play": False, "sigma": None})
        else:
            raise Exception(f"framework {framework} is not supported yet.")
