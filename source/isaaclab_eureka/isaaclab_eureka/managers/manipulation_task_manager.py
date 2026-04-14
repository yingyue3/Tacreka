# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

"""Task manager for IsaacLab manager-based manipulation environments.

Unlike :class:`EurekaTaskManager` and :class:`TacrekaTaskManager`, which target
``DirectRLEnv`` (where rewards flow through ``_get_rewards``), this manager is
designed for **``ManagerBasedRLEnv``** tasks such as ``Isaac-Lift-Cube-Franka-v0``.

Key architectural differences
------------------------------
* **Reward injection**: wraps ``env.reward_manager.compute()`` instead of patching
  ``env._get_rewards``.  The original reward terms remain active as an *oracle*
  signal; the LLM-generated reward replaces the training signal.
* **Observation description**: because ``ManagerBasedRLEnv`` has no ``_get_observations``
  method, the observation string is synthesised from the env config's
  ``ObservationsCfg`` dataclass and known scene-entity access patterns.
* **LLM reward signature**: identical to the existing convention —
  ``def _get_rewards_eureka(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]``
  bound to the env instance — but ``self`` is a ``ManagerBasedRLEnv``, so the LLM
  must access state via ``self.scene["name"].data.*``.
"""

import dataclasses
import inspect
import math
import multiprocessing
import os
import traceback
import types
from contextlib import nullcontext
from datetime import datetime
from typing import Literal

from isaaclab_eureka.learning_curve_utils import export_learning_curve_artifacts, resolve_checkpoint_path
from isaaclab_eureka.utils import MuteOutput, get_freest_gpu

# ---------------------------------------------------------------------------
# Template: wraps reward_manager.compute() on a ManagerBasedRLEnv.
#
# self_rm  = the RewardManager instance (bound via types.MethodType)
# self_rm._env = the ManagerBasedRLEnv
#
# Behaviour each step:
#   1. Run all original reward terms (oracle) for tracking / logging.
#   2. Call env._get_rewards_eureka() for the LLM reward.
#   3. Accumulate both into _eureka_episode_sums on the env.
#   4. Return the LLM reward as the RL training signal.
# ---------------------------------------------------------------------------
TEMPLATE_REWARD_WRAP_STRING = """
import torch

def _compute_with_eureka(self_rm, dt):
    oracle_reward = self_rm._compute_oracle(dt)
    env = self_rm._env
    eureka_reward, rewards_dict = env._get_rewards_eureka()
    env._eureka_episode_sums["oracle_total_rewards"] += oracle_reward
    env._eureka_episode_sums["eureka_total_rewards"] += eureka_reward
    for key in rewards_dict.keys():
        if key not in env._eureka_episode_sums:
            env._eureka_episode_sums[key] = torch.zeros(env.num_envs, device=env.device)
        env._eureka_episode_sums[key] += rewards_dict[key]
    return eureka_reward
"""

# ---------------------------------------------------------------------------
# Template: patches _reset_idx on a ManagerBasedRLEnv.
#
# Mirrors the pattern in EurekaTaskManager / TacrekaTaskManager:
#   1. Evaluate the success_metric expression BEFORE original reset so that
#      scene buffers still hold the episode-end state.
#   2. Call _reset_idx_original (which resets managers, buffers, etc.).
#   3. Merge per-episode Eureka averages into self.extras["log"] and zero sums.
#
# {success_metric} is replaced with the full assignment statement
#   extras['Eureka/success_metric'] = <expression>
# or left empty if no success metric is configured.
# ---------------------------------------------------------------------------
TEMPLATE_RESET_STRING = """
import torch

@torch.inference_mode()
def _reset_idx(self, env_ids):
    if env_ids is None or len(env_ids) == self.num_envs:
        env_ids = torch.arange(self.num_envs, device=self.device)
    extras = dict()
    # Evaluate success metric before the original reset so scene buffers reflect
    # the episode-end state (they will be overwritten by _reset_idx_original).
    {success_metric}
    self._reset_idx_original(env_ids)
    if "log" not in self.extras:
        self.extras["log"] = dict()
    for key in self._eureka_episode_sums.keys():
        episodic_sum_avg = torch.mean(self._eureka_episode_sums[key][env_ids])
        extras["Eureka/" + key] = episodic_sum_avg / self.max_episode_length_s
        self._eureka_episode_sums[key][env_ids] = 0.0
    self.extras["log"].update(extras)
"""

# ---------------------------------------------------------------------------
# Imports prepended to every LLM-generated reward function string before exec.
# Provides common utilities without requiring the LLM to write import lines.
# ---------------------------------------------------------------------------
_LLM_REWARD_IMPORT_PREFIX = (
    "import torch\n"
    "from isaaclab.utils.math import (\n"
    "    combine_frame_transforms,\n"
    "    subtract_frame_transforms,\n"
    "    quat_rotate,\n"
    "    quat_rotate_inverse,\n"
    ")\n"
    "from isaaclab.assets import RigidObject\n"
    "from isaaclab.sensors import FrameTransformer\n"
    "\n"
)


class ManipulationTaskManager:
    """Task manager for IsaacLab ``ManagerBasedRLEnv`` manipulation tasks.

    Supports Eureka-style LLM reward injection for tasks registered with
    ``entry_point="isaaclab.envs:ManagerBasedRLEnv"`` (e.g.
    ``Isaac-Lift-Cube-Franka-v0``).

    Usage::

        manager = ManipulationTaskManager(
            task="Isaac-Lift-Cube-Franka-v0",
            success_metric_string=(
                "torch.linalg.norm("
                "(self.scene['robot'].data.root_pos_w[env_ids]"
                " + self.command_manager.get_command('object_pose')[env_ids, :3])"
                " - self.scene['object'].data.root_pos_w[env_ids, :3],"
                " dim=1).mean()"
            ),
        )
        obs_string = manager.get_observations_method_as_string
        # ... feed obs_string to LLM, receive reward_fn_string ...
        results = manager.train([reward_fn_string])
        manager.close()

    LLM reward function contract
    -----------------------------
    The LLM must generate a function with this exact signature::

        def _get_rewards_eureka(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            # self  : ManagerBasedRLEnv
            # Scene access (examples):
            #   cube_pos   = self.scene["object"].data.root_pos_w          # (N, 3)
            #   ee_pos     = self.scene["ee_frame"].data.target_pos_w[..., 0, :]  # (N, 3)
            #   goal_robot = self.command_manager.get_command("object_pose")[:, :3]  # (N, 3)
            #   robot_pos  = self.scene["robot"].data.root_pos_w            # (N, 3)
            #   goal_world = robot_pos + goal_robot  # valid (identity base rotation)
            total_reward = ...          # shape (num_envs,)
            reward_dict  = {"term": ...}  # named components for logging
            return total_reward, reward_dict
    """

    def __init__(
        self,
        task: str,
        rl_library: Literal["rsl_rl", "rl_games"] = "rsl_rl",
        num_processes: int = 1,
        device: str = "cuda",
        env_seed: int = 42,
        max_training_iterations: int = 100,
        success_metric_string: str = "",
        log_namespace: str = "manipulation_eureka",
        rl_log_root_dir: str | None = None,
    ):
        """Initialise the task manager and start worker processes.

        Args:
            task: Gym task id, e.g. ``"Isaac-Lift-Cube-Franka-v0"``.
            rl_library: ``"rsl_rl"`` (default) or ``"rl_games"``.
            num_processes: Number of parallel training runs (one process each).
            device: ``"cuda"`` (auto-selects the freest GPU) or e.g. ``"cuda:0"``.
            env_seed: Seed for the simulation environment.
            max_training_iterations: PPO iterations per training call.
            success_metric_string: Python expression (as a string) evaluated
                inside the patched ``_reset_idx`` that computes a scalar
                success signal.  Must be valid in a context where ``self`` is
                the env and ``env_ids`` is a 1-D int tensor.
            log_namespace: Sub-folder name under ``logs/rl_runs/rsl_rl_<ns>/``.
            rl_log_root_dir: Override the entire log root directory.
        """
        self._task = task
        self._rl_library = rl_library
        self._num_processes = num_processes
        self._device = device
        self._max_training_iterations = max_training_iterations
        self._success_metric_string = success_metric_string
        self._env_seed = env_seed
        self._rl_log_root_dir = os.path.abspath(rl_log_root_dir) if rl_log_root_dir else None
        sanitized = str(log_namespace).strip().replace(" ", "_")
        self._log_namespace = sanitized if sanitized else "manipulation_eureka"

        # Wrap success_metric string into a full assignment statement.
        if self._success_metric_string:
            self._success_metric_string = (
                "extras['Eureka/success_metric'] = " + self._success_metric_string
            )

        self._processes: dict[int, multiprocessing.Process] = {}
        # Per-worker queue: main → worker (reward function strings)
        self._rewards_queues = [multiprocessing.Queue() for _ in range(self._num_processes)]
        # Worker 0 → main: observation description string
        self._observations_queue: multiprocessing.Queue = multiprocessing.Queue()
        # Workers → main: training results
        self._results_queue: multiprocessing.Queue = multiprocessing.Queue()
        self.termination_event = multiprocessing.Event()

        for idx in range(self._num_processes):
            p = multiprocessing.Process(target=self._worker, args=(idx, self._rewards_queues[idx]))
            self._processes[idx] = p
            p.start()

        # Block until worker 0 sends the observation description.
        self._get_observations_as_string = self._observations_queue.get()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def get_observations_method_as_string(self) -> str:
        """Human-readable description of available observations and scene access.

        This string is generated from the env config's ``ObservationsCfg``
        (observation term function source code) plus documented scene-entity
        access patterns for the LLM.
        """
        return self._get_observations_as_string

    def close(self):
        """Shut down all worker processes and close simulators."""
        self.termination_event.set()
        for q in self._rewards_queues:
            q.put("Stop")
        for p in self._processes.values():
            p.join()

    def train(self, get_rewards_method_as_string: list[str]) -> list[dict]:
        """Train with LLM-generated reward functions.

        Each string must be a Python function definition starting with
        ``def _get_rewards_eureka(self):``.

        Args:
            get_rewards_method_as_string: One reward function per process.

        Returns:
            List of result dicts, one per process, with keys:
            ``"success"``, ``"log_dir"``, ``"run_dir"``,
            ``"checkpoint_file"`` (if produced), ``"exception"`` (if failed),
            ``"learning_curve"`` (if exported successfully).
        """
        if len(get_rewards_method_as_string) != self._num_processes:
            raise ValueError(
                f"Expected {self._num_processes} reward strings, "
                f"got {len(get_rewards_method_as_string)}."
            )
        for q, reward_str in zip(self._rewards_queues, get_rewards_method_as_string):
            q.put(reward_str)

        results = [None] * self._num_processes
        for _ in range(self._num_processes):
            idx, result = self._results_queue.get()
            results[idx] = result
        return results

    # ------------------------------------------------------------------
    # Worker internals (run inside child processes)
    # ------------------------------------------------------------------

    def _worker(self, idx: int, rewards_queue: multiprocessing.Queue):
        """Entry point for each training process."""
        self._idx = idx
        while not self.termination_event.is_set():
            if not hasattr(self, "_env"):
                self._create_environment()
                if self._idx == 0 and not hasattr(self, "_observation_string"):
                    self._observation_string = self._build_observation_string()
                    self._observations_queue.put(self._observation_string)

            reward_func_string = rewards_queue.get()
            if (
                isinstance(reward_func_string, str)
                and reward_func_string.startswith("def _get_rewards_eureka(self)")
            ):
                try:
                    self._prepare_eureka_environment(reward_func_string)
                    context = MuteOutput() if self._idx > 0 else nullcontext()
                    with context:
                        self._run_training()
                    result = {"success": True, "log_dir": self._log_dir, "run_dir": self._run_dir}
                    checkpoint_file = resolve_checkpoint_path(self._run_dir)
                    if checkpoint_file is not None:
                        result["checkpoint_file"] = checkpoint_file
                    try:
                        lc_dir = os.path.join(self._run_dir, "learning_curves")
                        lc = export_learning_curve_artifacts(
                            self._log_dir,
                            output_dir=lc_dir,
                            run_name=os.path.basename(self._run_dir),
                        )
                        if lc is not None:
                            result["learning_curve"] = lc
                    except Exception as plot_err:
                        result["learning_curve_error"] = str(plot_err)
                except Exception as e:
                    result = {"success": False, "exception": str(e)}
                    print(traceback.format_exc())
            else:
                result = {
                    "success": False,
                    "exception": (
                        "Reward function must start with 'def _get_rewards_eureka(self)'."
                    ),
                }

            self._results_queue.put((self._idx, result))

        print(f"[ManipulationTaskManager] Worker {self._idx} terminated.")
        self._env.close()
        self._simulation_app.close()

    def _create_environment(self):
        """Create a ``ManagerBasedRLEnv`` for the configured task."""
        from isaaclab.app import AppLauncher

        if self._device == "cuda":
            device_id = get_freest_gpu()
            self._device = f"cuda:{device_id}"

        app_launcher = AppLauncher(headless=True, device=self._device)
        self._simulation_app = app_launcher.app

        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401
        from isaaclab.envs import ManagerBasedRLEnvCfg
        from isaaclab_tasks.utils import parse_env_cfg

        env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(self._task)
        env_cfg.sim.device = self._device
        env_cfg.seed = self._env_seed
        self._env = gym.make(self._task, cfg=env_cfg)

    def _build_observation_string(self) -> str:
        """Build an observation description string for the LLM.

        For ``ManagerBasedRLEnv`` there is no ``_get_observations`` method to
        inspect.  Instead this method:

        1. Iterates over the env config's ``ObservationsCfg`` dataclass and
           extracts the source code of every observation term function.
        2. Appends documented scene-entity access patterns that the LLM can
           use directly when writing reward functions.

        Returns:
            A single multi-line string for use as the ``observations`` context
            in the LLM prompt.
        """
        env = self._env.unwrapped
        obs_cfg = env.cfg.observations
        lines = [
            "# ======================================================================",
            "# ManagerBasedRLEnv — Observation Terms & Scene Access Reference",
            "#",
            "# The LLM reward function receives `self` (the ManagerBasedRLEnv).",
            "# Access scene entities with  self.scene['name'].data.*",
            "# ======================================================================",
            "",
        ]

        # --- Observation term function source code ---
        if dataclasses.is_dataclass(obs_cfg):
            for group_field in dataclasses.fields(obs_cfg):
                group = getattr(obs_cfg, group_field.name, None)
                if group is None or not dataclasses.is_dataclass(group):
                    continue
                lines.append(f"# === Observation group: {group_field.name!r} ===")
                for term_field in dataclasses.fields(group):
                    term = getattr(group, term_field.name, None)
                    if term is None or not hasattr(term, "func"):
                        continue
                    lines.append(f"# Term '{term_field.name}':")
                    try:
                        src = inspect.getsource(term.func)
                        lines.append(src)
                    except (OSError, TypeError):
                        lines.append(f"#   {term.func}")
                lines.append("")

        # --- Scene entity access patterns ---
        lines += [
            "# --- Key scene entity access patterns (use in reward function) ---",
            "#",
            "# Object (cube) state:",
            "#   self.scene['object'].data.root_pos_w          (num_envs, 3)   world pos",
            "#   self.scene['object'].data.root_lin_vel_w       (num_envs, 3)   world linear vel",
            "#   self.scene['object'].data.root_ang_vel_w       (num_envs, 3)   world angular vel",
            "#   self.scene['object'].data.root_quat_w          (num_envs, 4)   world orientation [w,x,y,z]",
            "#",
            "# Robot state:",
            "#   self.scene['robot'].data.root_pos_w            (num_envs, 3)   robot base world pos",
            "#   self.scene['robot'].data.root_quat_w           (num_envs, 4)   robot base world orientation",
            "#   self.scene['robot'].data.joint_pos             (num_envs, J)   joint positions",
            "#   self.scene['robot'].data.joint_vel             (num_envs, J)   joint velocities",
            "#   self.scene['robot'].data.joint_pos_target      (num_envs, J)   commanded joint targets",
            "#",
            "# End-effector frame (Franka panda_hand + 0.1034 m offset):",
            "#   self.scene['ee_frame'].data.target_pos_w[..., 0, :]   (num_envs, 3)  EE world pos",
            "#   self.scene['ee_frame'].data.target_quat_w[..., 0, :]  (num_envs, 4)  EE world orientation",
            "#",
            "# Goal command ('object_pose' — resampled every 5 s / full episode):",
            "#   self.command_manager.get_command('object_pose')[:, :3]",
            "#       (num_envs, 3) desired cube position in robot-base frame",
            "#   Goal in world frame (Franka base has identity rotation):",
            "#       goal_world = self.scene['robot'].data.root_pos_w + cmd[:, :3]",
            "#",
            "# Episode / env metadata:",
            "#   self.episode_length_buf   (num_envs,)  current step within episode",
            "#   self.max_episode_length   int          max steps per episode",
            "#   self.num_envs             int          number of parallel envs",
            "#   self.device               str          e.g. 'cuda:0'",
            "#",
            "# Math utilities (already imported in reward function namespace):",
            "#   torch.linalg.norm(v, dim=1)                        per-env L2 norm",
            "#   torch.tanh(x), torch.where(cond, a, b)             elementwise ops",
            "#   combine_frame_transforms(t_ab, q_ab, t_bc, q_bc)  compose SE3",
            "#   subtract_frame_transforms(t_ab, q_ab, t_ac)       express c in b",
            "#   quat_rotate(q, v), quat_rotate_inverse(q, v)      rotate vectors",
        ]
        return "\n".join(lines)

    def _prepare_eureka_environment(self, get_rewards_method_as_string: str):
        """Inject the LLM reward into the environment.

        First call (per worker lifetime):
        * Saves ``reward_manager.compute`` as ``_compute_oracle``.
        * Installs ``_compute_with_eureka`` as the new ``reward_manager.compute``.
        * Saves ``env._reset_idx`` as ``env._reset_idx_original``.
        * Installs the patched ``_reset_idx`` that logs episodic sums.

        Every call:
        * Binds the new LLM reward function as ``env._get_rewards_eureka``.
        * Re-initialises ``_eureka_episode_sums`` buffers.
        """
        import torch

        env = self._env.unwrapped
        reward_manager = env.reward_manager
        namespace: dict = {}

        if not hasattr(env, "_get_rewards_eureka"):
            # --- 1. Wrap reward_manager.compute() ---
            reward_manager._compute_oracle = reward_manager.compute
            exec(TEMPLATE_REWARD_WRAP_STRING, namespace)
            reward_manager.compute = types.MethodType(
                namespace["_compute_with_eureka"], reward_manager
            )

            # --- 2. Patch _reset_idx ---
            env._reset_idx_original = env._reset_idx
            reset_string = TEMPLATE_RESET_STRING.format(
                success_metric=self._success_metric_string
            )
            if self._rl_library == "rl_games":
                reset_string = reset_string.replace("@torch.inference_mode()", "")
            exec(reset_string, namespace)
            env._reset_idx = types.MethodType(namespace["_reset_idx"], env)

        # --- 3. Bind LLM reward function ---
        full_reward_string = _LLM_REWARD_IMPORT_PREFIX + get_rewards_method_as_string
        exec(full_reward_string, namespace)
        env._get_rewards_eureka = types.MethodType(namespace["_get_rewards_eureka"], env)

        # --- 4. (Re-)initialise episodic sum buffers ---
        env._eureka_episode_sums = {
            "eureka_total_rewards": torch.zeros(env.num_envs, device=env.device),
            "oracle_total_rewards": torch.zeros(env.num_envs, device=env.device),
        }

    def _run_training(self):
        """Launch the RL training loop (rsl_rl or rl_games) inside the worker."""
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        if self._rl_library == "rsl_rl":
            self._run_training_rsl_rl()
        elif self._rl_library == "rl_games":
            self._run_training_rl_games()
        else:
            raise ValueError(f"Unsupported RL library: {self._rl_library!r}")

    def _run_training_rsl_rl(self):
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner

        agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(
            self._task, "rsl_rl_cfg_entry_point"
        )
        agent_cfg.device = self._device
        agent_cfg.max_iterations = self._max_training_iterations

        if self._rl_log_root_dir:
            log_root_path = self._rl_log_root_dir
        else:
            log_root_path = os.path.join(
                "logs", "rl_runs", f"rsl_rl_{self._log_namespace}", agent_cfg.experiment_name
            )
        log_root_path = os.path.abspath(log_root_path)
        os.makedirs(log_root_path, exist_ok=True)
        print(f"[ManipulationTaskManager] Logging to: {log_root_path}")

        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_Run-{self._idx}"
        if self._rl_log_root_dir:
            exp_name = str(agent_cfg.experiment_name).replace(os.sep, "_").replace(" ", "_")
            log_dir = f"rsl_rl_{exp_name}_{log_dir}"
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"

        self._run_dir = os.path.join(log_root_path, log_dir)
        self._log_dir = self._run_dir

        env = RslRlVecEnvWrapper(self._env)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=self._log_dir, device=agent_cfg.device)
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    def _run_training_rl_games(self):
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
        from rl_games.common import env_configurations, vecenv
        from rl_games.common.algo_observer import IsaacAlgoObserver
        from rl_games.torch_runner import Runner

        agent_cfg = load_cfg_from_registry(self._task, "rl_games_cfg_entry_point")
        agent_cfg["params"]["config"]["max_epochs"] = self._max_training_iterations
        agent_cfg["params"]["config"]["device"] = self._device
        agent_cfg["params"]["config"]["device_name"] = self._device

        if self._rl_log_root_dir:
            log_root_path = self._rl_log_root_dir
        else:
            log_root_path = os.path.join(
                "logs", "rl_runs", f"rl_games_{self._log_namespace}",
                agent_cfg["params"]["config"]["name"],
            )
        log_root_path = os.path.abspath(log_root_path)
        os.makedirs(log_root_path, exist_ok=True)
        print(f"[ManipulationTaskManager] Logging to: {log_root_path}")

        log_dir = (
            agent_cfg["params"]["config"].get(
                "full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            )
            + f"_Run-{self._idx}"
        )
        if self._rl_log_root_dir:
            exp_name = (
                str(agent_cfg["params"]["config"]["name"]).replace(os.sep, "_").replace(" ", "_")
            )
            log_dir = f"rl_games_{exp_name}_{log_dir}"

        agent_cfg["params"]["config"]["train_dir"] = log_root_path
        agent_cfg["params"]["config"]["full_experiment_name"] = log_dir
        self._run_dir = os.path.join(log_root_path, log_dir)
        self._log_dir = os.path.join(self._run_dir, "summaries")

        clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
        clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
        env = RlGamesVecEnvWrapper(self._env, self._device, clip_obs, clip_actions)

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
