# This version is updated to use the new feature generation prompt and reward generation prompt.
# 
# Update by Yingyue Cao on 03/02/2026
#
# Feature update: 
# 1. Use the schema to generate the feature generation prompt.
# 2. keep the human liked features, and propose two alternative features: one is explore mode, one is exploit mode.
# 3. Reduce the number of features to generate to 1-4.


import datetime
import os
import json
from typing import Literal

# we import this here to avoid GLIBCXX_3.4.30 error in Isaac Sim 5.1
from isaaclab.app import AppLauncher
from isaaclab_eureka import EUREKA_ROOT_DIR
from isaaclab_eureka.learning_curve_utils import export_learning_curve_artifacts, resolve_checkpoint_path
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
    FEATURE_GEN_EXPLORE_FEEDBACK_PROMPT,
    FEATURE_GEN_EXPLOIT_FEEDBACK_PROMPT,
    FEATURE_AS_ONE_FAILURE_FEEDBACK_PROMPT,
    FEATURE_AS_ONE_SUCCESS_POST_FEEDBACK_PROMPT,
    FEATURE_AS_ONE_SUCCESS_PRE_FEEDBACK_PROMPT,
    TASK_SUCCESS_PRE_FEEDBACK_PROMPT,
    FEATURE_AS_ONE_NEW_FEATURES_FEEDBACK_PROMPT
)
from isaaclab_eureka.managers import EurekaTaskManager, LLMManagerTac, RecordManagerQuad, VideoIsaac, ManipulationTaskManager
from isaaclab_eureka.utils import load_tensorboard_logs
from isaaclab_eureka.managers.feedback_manager import (
    HumanFeedbackManager, 
    RewardInfo, 
    FeatureSpec
)



class Tacreka_Preference:
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
        human_feedback: bool = True,
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
        self.local_training = True
        if task == "Isaac-Lift-Cube-Franka-v0":
            self.manipulation_task = True
        else:
            self.manipulation_task = False


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
        # num processes is the number of parallel runs for the LLM (reward components number)
        self._num_processes = num_parallel_runs
        # num parallel runs is the number of parallel runs for the task (reward functions number)
        self._num_parallel_runs = 3

        # Logging
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._log_dir = os.path.join(EUREKA_ROOT_DIR, "logs", "tacreka_sr", task, timestamp)
        self._rl_runs_dir = os.path.join(self._log_dir, "rl_runs")
        os.makedirs(self._log_dir)

        # Persistent trajectory store + human-preference dataset for post-hoc TAC.
        # The video files under ./ratings/ are overwritten between iterations, so we
        # save each rollout's per-step trajectory to a unique path under the log dir
        # and keep a video-path -> trajectory-path map that we update in lockstep
        # with the existing rename pattern.
        self._traj_dir = os.path.join(self._log_dir, "trajectories")
        os.makedirs(self._traj_dir, exist_ok=True)
        self._human_pref_jsonl_path = os.path.join(self._log_dir, "human_preferences.jsonl")
        self._video_to_traj: dict[str, str] = {}
        self._traj_metadata: dict[str, dict] = {}
        self._traj_seq_counter: int = 0

        print("[INFO]: Setting up the LLM Manager...")
        self._llm_manager = LLMManagerTac(
            gpt_model=gpt_model,
            temperature=temperature,
            system_prompt=FEATURE_AS_ONE_REWARD_INITIAL_PROMPT,
            feature_prompt=FEATURE_GEN_INITIAL_PROMPT,
        )

        print("[INFO]: Setting up the Task Manager...")
        if self.manipulation_task:
            self._task_manager = ManipulationTaskManager(
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

        print("[INFO]: Setting up the Record Manager...")
        if not self.manipulation_task:
            self._record_manager = RecordManagerQuad(
                task=task,
                num_envs=1,
                device=device,
                max_frames=900,
                num_episodes=1,
                save_trajectory = True,
                trajectory_gamma= 1.0,
            )
        else:
            self._record_manager = VideoIsaac(
                task=task,
                device=device,
            )
        

        print("[INFO]: Setting up the Feedback Manager...")
        self._feedback_manager = HumanFeedbackManager(port=8889)

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
        feature_gen_prompt = FEATURE_GEN_PROMPT.format(
            task_description=self._task_description,
            # success_metric_to_win=self._success_metric_to_win,
            get_observations_method_as_string=self._task_manager.get_observations_method_as_string,
        )
        # The assistant prompt is used to feed the previous LLM output back into the LLM
        assistant_prompt = None
        rw_gen_assistant_prompt = None
        rw_gen_user_prompt = None

        # The best run across all iterations
        best_run_results = {"success_metric": None}

        for iter in range(max_eureka_iterations):
            print(f"\n{'#' * 20} Running Eureka Iteration {iter} {'#' * 20} \n")
            # Generate the GPT reward methods
            # Assistant prompt is the previous generated feature decomposition, user prompt is feedback from the previous iteration
            if assistant_prompt is None:
                feature_gen_outputs = self._llm_manager.feature_gen(user_prompt=feature_gen_prompt, assistant_prompt=assistant_prompt, num_suggestion= 3)
                feature_strings = feature_gen_outputs["feature_strings"]
                print(f"\n{'+' * 20} {len(feature_strings)} Features Generated {'+' * 20} \n")
            else:
                feature_gen_outputs_explore = self._llm_manager.feature_gen(user_prompt=feature_gen_prompt + FEATURE_GEN_EXPLORE_FEEDBACK_PROMPT, assistant_prompt=assistant_prompt, num_suggestion=3)
                # feature_gen_outputs_exploit = self._llm_manager.feature_gen(user_prompt=feature_gen_prompt + FEATURE_GEN_EXPLOIT_FEEDBACK_PROMPT, assistant_prompt=assistant_prompt, num_suggestion=1)
                feature_gen_outputs["raw_outputs"] = feature_gen_outputs_explore["raw_outputs"]
                feature_gen_outputs["feature_strings"] = feature_gen_outputs_explore["feature_strings"]
                feature_strings = feature_gen_outputs["feature_strings"]
                print(f"\n{'+' * 20} 1 Feature Reused, 2 Features Generated {'+' * 20} \n")
            # else:
            #     print(f"\n{'+' * 20} All Features Reused {'+' * 20} \n")
            # self._llm_manager.single_feature_reset()
            llm_outputs = []
            gpt_reward_method_strings = []
            for idx, feature_string in enumerate(feature_strings):
                if rw_gen_user_prompt is None:
                    rw_gen_user_prompt_iter = FEATURE_AS_ONE_REWARD_PROMPT.format(
                        task_description=self._task_description,
                        # success_metric_to_win=self._success_metric_to_win,
                        get_observations_method_as_string=self._task_manager.get_observations_method_as_string,
                        FEATURES_JSON=feature_string,
                    )
                elif feature_gen_prompt != "N" and rw_gen_assistant_prompt is not None:
                    # rw_gen_user_prompt_iter = FEATURE_AS_ONE_NEW_FEATURES_FEEDBACK_PROMPT.format(FEATURES_JSON=feature_string)
                    # rw_gen_user_prompt_iter = rw_gen_user_prompt + FEATURE_AS_ONE_SUCCESS_POST_FEEDBACK_PROMPT.format(FEATURES_JSON=feature_string)
                    rw_gen_user_prompt_iter = FEATURE_AS_ONE_REWARD_PROMPT.format(
                        task_description=self._task_description,
                        success_metric_to_win=self._success_metric_to_win,
                        get_observations_method_as_string=self._task_manager.get_observations_method_as_string,
                        FEATURES_JSON=feature_string,
                    ) + FEATURE_AS_ONE_NEW_FEATURES_FEEDBACK_PROMPT
                else:
                    rw_gen_user_prompt_iter = rw_gen_user_prompt
                reward_code = self._llm_manager.single_feature_prompt(user_prompt=rw_gen_user_prompt_iter, assistant_prompt=rw_gen_assistant_prompt, 
                num_suggestion= 1,
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
                    # print(f"feature_idx: {idx}")
                # print(f"inner loop {i} of {len(llm_outputs)}")
            # print(f"outer loop {i} of {len(llm_outputs)}")
            # Train the RL agent
            results = []
            reward_strings = []
            for llm_output in llm_outputs:
                reward_strings += llm_output["reward_strings"]
            print("+"*10 + " Training Started" + "+"*10)
            if not self.local_training:
                results = self._task_manager.train(reward_strings)
            else:
                for reward_string in reward_strings:
                    results += self._task_manager.train([reward_string])
            # results = self._task_manager.train(reward_strings)
            # Give TensorBoard time to flush logs before reading them
            import time
            time.sleep(1.0)  # Wait 1 second for TensorBoard to flush
            # Evaluate the results
            iter_best_success_metric = None
            best_run_idx = 0
            best_reward_components = 0
            best_run_feature_components = None
            best_run_checkpoint = None
            print("+"*10 + " Training Ends, Evaluating Results" + "+"*10)
            for idx, result in enumerate(results):
                feedback_result = None
                checkpoint_list = []

                # Human provide feedback for the best feature sets
                feature_idx = gpt_reward_method_strings[idx]["feature_idx"]
                results[idx]["reward_components"] = feature_gen_outputs["feature_strings"][feature_idx]
                            
                # Human provide feedback for videos     
                if not result["success"]:
                    checkpoint_list.append("NONE")
                    user_feedback_prompt_rw_gen = FEATURE_AS_ONE_FAILURE_FEEDBACK_PROMPT.format(traceback_msg=result["exception"])
                    user_feedback_prompt = "N"
                    success_metric_max = None
                    print("Failed to generate correct reward function, no video recorded.")
                    if iter_best_success_metric is None:
                        # best_run_checkpoint = "NONE"
                        new_run_checkpoint = "NONE"
                    else:
                        new_run_checkpoint = "NONE"

                else:
                    # Compute the performance metrics
                    print("Successfully trained the reward function, generating videos")
                    eureka_task_feedback, success_metric_max, rewards_correlation, oracle_reward = self._get_eureka_task_feedback(
                        result["log_dir"], self._feedback_subsampling
                    )

                    # Generate the user feedback prompt
                    user_feedback_prompt = FEATURE_GEN_FEEDBACK_PROMPT + eureka_task_feedback
                        

                    user_feedback_prompt_rw_gen = (
                        FEATURE_AS_ONE_SUCCESS_PRE_FEEDBACK_PROMPT.format(feedback_subsampling=self._feedback_subsampling)
                        + eureka_task_feedback
                    )

                    # Store the results
                    results[idx]["eureka_task_feedback"] = eureka_task_feedback
                    results[idx]["success_metric_max"] = success_metric_max
                    results[idx]["reward_correlation"] = rewards_correlation
                    results[idx]["oracle_reward"] = oracle_reward
                    results[idx]["reward_strings"] = reward_strings
                    results[idx]["training_log_dir"] = result.get("log_dir")
                    results[idx]["training_run_dir"] = result.get("run_dir", result.get("log_dir"))
                    results[idx]["checkpoint_file"] = result.get("checkpoint_file") or resolve_checkpoint_path(
                                    result.get("run_dir", result.get("log_dir"))
                                )
                    results[idx]["learning_curve"] = result.get("learning_curve")
                    # Check the best performing metric, determined by the minimum distance from the win target
                    if success_metric_max is not None:
                        if iter_best_success_metric is None:
                            iter_best_success_metric = success_metric_max
                        elif abs(iter_best_success_metric - self._success_metric_to_win) > abs(success_metric_max - self._success_metric_to_win):
                            iter_best_success_metric = success_metric_max
                        new_run_checkpoint = result.get("checkpoint_file") or resolve_checkpoint_path(
                            result.get("run_dir", result["log_dir"])
                        )
                        print(f"[INFO] new_run_checkpoint: {new_run_checkpoint}")
                        self._record_and_track(
                            video_output="./ratings/run_1.mp4",
                            checkpoint=new_run_checkpoint,
                            iter_idx=iter,
                            role="run_1",
                            candidate_source=gpt_reward_method_strings[idx]["reward_strings"],
                            feature_components=feature_gen_outputs["feature_strings"][feature_idx],
                            feature_idx=feature_idx,
                            result_idx=idx,
                        )
                        checkpoint_list.append("./ratings/run_1.mp4")
                    #     if best_run_results["success_metric"] is None or (
                    #     np.abs(iter_best_success_metric - self._success_metric_to_win)
                    #     < np.abs(best_run_results["success_metric"] - self._success_metric_to_win)
                    #     ):
                    #             best_run_results["success_metric"] = iter_best_success_metric
                    #             best_run_results["oracle_reward"] = oracle_reward
                    #             best_run_results["reward_correlation"] = rewards_correlation
                    #             best_run_results["task_feedback"] = eureka_task_feedback
                    #             best_run_results["feature_idx"] = gpt_reward_method_strings[idx]["feature_idx"]
                    #             best_run_feature_idx = gpt_reward_method_strings[idx]["feature_idx"]
                    #             best_run_results["feature_components"] = feature_gen_outputs["raw_outputs"][best_run_feature_idx]
                    #             best_run_results["gpt_reward_method"] = gpt_reward_method_strings[idx]["reward_strings"]
                    #             best_run_results["training_log_dir"] = result.get("log_dir")
                    #             best_run_results["training_run_dir"] = result.get("run_dir", result.get("log_dir"))
                    #             best_run_results["checkpoint_file"] = result.get("checkpoint_file") or resolve_checkpoint_path(
                    #                 result.get("run_dir", result.get("log_dir"))
                    #             )
                    #             best_run_results["learning_curve"] = result.get("learning_curve")
                    #             print("logging best metric")

                if best_run_feature_components is None:
                    best_run_feature_idx = gpt_reward_method_strings[idx]["feature_idx"]
                    best_run_feature_components = feature_gen_outputs["feature_strings"][best_run_feature_idx]
                    best_run_checkpoint = new_run_checkpoint
                    if best_run_checkpoint != "NONE":
                        self._rename_video("./ratings/run_1.mp4", "./ratings/run_2.mp4")
                else:
                    reward_info_v1 = RewardInfo(
                    name="Run 1",
                    feature_specs=feature_gen_outputs["feature_strings"][feature_idx],
                    )
                    reward_info_v2 = RewardInfo(
                    name="Run 2",
                    feature_specs=best_run_feature_components)
                    feedback_result = self._feedback_manager.select_video(
                        task_description=self._task_description,
                        reward_infos=[reward_info_v1, reward_info_v2],
                        allow_text_feedback=False,
                        allow_rating=False,
                    )

                    print("Video Feedback Gathering Process Started...")
                
                    if best_run_checkpoint == "NONE":
                        checkpoint_list.append("NONE")
                    else:
                        checkpoint_list.append("./ratings/run_2.mp4")
                    feedback_result = self._feedback_manager.select_video(
                        video_paths=checkpoint_list,
                        task_description="Now provided with videos of the two reward sets, please revise your preference on the best feature sets",
                        reward_infos=[reward_info_v1, reward_info_v2],
                        allow_text_feedback=False,
                        allow_rating=False,
                    )
                    self._log_human_preference(
                        iter_idx=iter,
                        context="iteration_internal",
                        video_paths=checkpoint_list,
                        winner_index=feedback_result.selected_index,
                        reward_infos=[reward_info_v1, reward_info_v2],
                    )
                    print(f"[INFO] Feedback result index: {feedback_result.selected_index}")
                    print(f"[INFO] Feedback result selected video: {feedback_result.selected_video}")
                    if feedback_result.selected_index == 0:
                        best_reward_components = idx
                        best_run_feature_idx = gpt_reward_method_strings[idx]["feature_idx"]
                        best_run_feature_components = feature_gen_outputs["feature_strings"][best_run_feature_idx]
                        # iter_best_success_metric = success_metric_max
                        best_run_idx = idx
                        if new_run_checkpoint != "NONE":
                            self._rename_video("./ratings/run_1.mp4", "./ratings/run_2.mp4")
                        best_run_checkpoint = new_run_checkpoint

                # Add the prompts
                feature_idx = gpt_reward_method_strings[idx]["feature_idx"]
                results[idx]["user_prompt"] = user_feedback_prompt
                results[idx]["assistant_prompt"] = feature_gen_outputs["raw_outputs"][feature_idx]
                results[idx]["user_prompt_rw_gen"] = user_feedback_prompt_rw_gen
                results[idx]["feature_idx"] = feature_idx
                results[idx]["assistant_prompt_rw_gen"] = gpt_reward_method_strings[idx]["raw_output"]
                results[idx]["gpt_reward_method"] = gpt_reward_method_strings[idx]["reward_strings"]
                results[idx]["feature_components"] = feature_gen_outputs["feature_strings"][feature_idx]
            
            print(f"Best reward components: {best_reward_components}")
            # print(f"Best run results: {results[best_reward_components]}")
            self._log_iteration_results(iter, results)

            # if (
            #     best_run_results["success_metric"] is not None
            #     and np.abs(best_run_results["success_metric"] - self._success_metric_to_win)
            #     < self._success_metric_tolerance
            # ):
            #     print(f"Task solved with success metric: {best_run_results['success_metric']}")
            #     break

            # if user_feedback_prompt == "N":
            #     iter -= 1
            #     continue

            if "user_prompt" not in best_run_results:
                best_run_results["user_prompt"] = results[best_reward_components]["user_prompt"]
                best_run_results["assistant_prompt"] = results[best_reward_components]["assistant_prompt"]
                best_run_results["user_prompt_rw_gen"] = results[best_reward_components]["user_prompt_rw_gen"]
                best_run_results["assistant_prompt_rw_gen"] = results[best_reward_components]["assistant_prompt_rw_gen"]
                best_run_results["feature_idx"] = gpt_reward_method_strings[best_reward_components]["feature_idx"]
                best_run_results["feature_components"] = results[best_reward_components]["feature_components"]
                best_run_results["gpt_reward_method"] = results[best_reward_components]["gpt_reward_method"]
                    
                if results[best_reward_components]["success"]:
                    print("Loading success metric: new best run")
                    best_run_results["success_metric"] = results[best_reward_components]["success_metric_max"]
                    best_run_results["oracle_reward"] = results[best_reward_components]["oracle_reward"]
                    best_run_results["reward_correlation"] = results[best_reward_components]["reward_correlation"]
                    best_run_results["task_feedback"] = results[best_reward_components]["eureka_task_feedback"] 
                    best_run_results["training_log_dir"] = results[best_reward_components]["training_log_dir"]
                    best_run_results["training_run_dir"] = results[best_reward_components]["training_run_dir"]
                    best_run_results["checkpoint_file"] = results[best_reward_components]["checkpoint_file"]
                    best_run_results["learning_curve"] = results[best_reward_components]["learning_curve"]
                    self._rename_video("./ratings/run_2.mp4", "./ratings/best_run.mp4")
            else:
                print("Best run comparison process started.")
                checkpoint_list = []
                if results[best_reward_components]["success"]:
                    checkpoint_list.append("./ratings/run_2.mp4")
                else:
                    checkpoint_list.append("NONE")
                if best_run_results["success_metric"] is not None:
                    checkpoint_list.append("./ratings/best_run.mp4")
                else:
                    checkpoint_list.append("NONE")
                print(f"Checkpoint list: {checkpoint_list}")
                reward_info_v1 = RewardInfo(
                    name="Run 1",
                    feature_specs=best_run_feature_components,
                    )
                reward_info_v2 = RewardInfo(
                    name="Run 2",
                    feature_specs=best_run_results["feature_components"])
                feedback_result = self._feedback_manager.select_video(
                        video_paths=checkpoint_list,
                        task_description="Now provided with videos of the two reward sets, please revise your preference on the best feature sets",
                        reward_infos=[reward_info_v1, reward_info_v2],
                        allow_text_feedback=False,
                        allow_rating=False,
                    )
                self._log_human_preference(
                    iter_idx=iter,
                    context="best_run_comparison",
                    video_paths=checkpoint_list,
                    winner_index=feedback_result.selected_index,
                    reward_infos=[reward_info_v1, reward_info_v2],
                )
                if feedback_result.selected_index == 0:
                    best_run_results["user_prompt"] = results[best_reward_components]["user_prompt"]
                    best_run_results["assistant_prompt"] = results[best_reward_components]["assistant_prompt"]
                    best_run_results["user_prompt_rw_gen"] = results[best_reward_components]["user_prompt_rw_gen"]
                    best_run_results["assistant_prompt_rw_gen"] = results[best_reward_components]["assistant_prompt_rw_gen"]
                    best_run_results["feature_idx"] = gpt_reward_method_strings[best_reward_components]["feature_idx"]
                    best_run_results["feature_components"] = results[best_reward_components]["feature_components"]
                    best_run_results["gpt_reward_method"] = results[best_reward_components]["gpt_reward_method"]
                    
                    if results[best_reward_components]["success"]:
                        print("Loading success metric: new best run")
                        best_run_results["success_metric"] = results[best_reward_components]["success_metric_max"]
                        best_run_results["oracle_reward"] = results[best_reward_components]["oracle_reward"]
                        best_run_results["reward_correlation"] = results[best_reward_components]["reward_correlation"]
                        best_run_results["task_feedback"] = results[best_reward_components]["eureka_task_feedback"] 
                        best_run_results["training_log_dir"] = results[best_reward_components]["training_log_dir"]
                        best_run_results["training_run_dir"] = results[best_reward_components]["training_run_dir"]
                        best_run_results["checkpoint_file"] = results[best_reward_components]["checkpoint_file"]
                        best_run_results["learning_curve"] = results[best_reward_components]["learning_curve"]
                        self._rename_video("./ratings/run_2.mp4", "./ratings/best_run.mp4")
            
            assistant_prompt = best_run_results["assistant_prompt"]
            feature_gen_prompt = best_run_results["user_prompt"]
            rw_gen_assistant_prompt = best_run_results["assistant_prompt_rw_gen"]
            rw_gen_user_prompt = best_run_results["user_prompt_rw_gen"]
        
        self._log_final_results(best_run_results)
        # Close the task manager
        self._task_manager.close()
        self._record_manager.close()
        

    # ------------------------------------------------------------------
    # Trajectory + human-preference logging (post-hoc TAC support)
    # ------------------------------------------------------------------

    def _record_and_track(
        self,
        *,
        video_output: str,
        checkpoint: str,
        iter_idx: int,
        role: str,
        candidate_source: str | None = None,
        feature_components=None,
        feature_idx: int | None = None,
        result_idx: int | None = None,
    ) -> str | None:
        """Run ``RecordManagerQuad.record`` while persisting the trajectory to a
        stable per-iteration path and registering it in the video->trajectory map.

        Returns the absolute trajectory ``.pt`` path, or ``None`` if recording or
        trajectory saving failed.
        """
        self._traj_seq_counter += 1
        traj_filename = f"iter{iter_idx:04d}_{role}_seq{self._traj_seq_counter:04d}.pt"
        traj_path = os.path.join(self._traj_dir, traj_filename)
        try:
            saved = self._record_manager.record(
                checkpoint=checkpoint,
                output_file=video_output,
                trajectory_output_file=traj_path,
            )
        except TypeError:
            # Older record() signatures without trajectory_output_file: fall back so we
            # don't break runs that use a different recorder (e.g. VideoIsaac).
            self._record_manager.record(checkpoint=checkpoint, output_file=video_output)
            saved = None

        video_abs = os.path.abspath(video_output)
        if saved:
            saved_abs = os.path.abspath(saved)
            self._video_to_traj[video_abs] = saved_abs
            self._traj_metadata[saved_abs] = {
                "iter": int(iter_idx),
                "role": role,
                "result_idx": result_idx,
                "feature_idx": feature_idx,
                "feature_components": feature_components,
                "candidate_reward_function": candidate_source,
                "video_at_record_time": video_abs,
                "checkpoint": checkpoint,
            }
        else:
            # Drop any stale mapping pointing at this video path.
            self._video_to_traj.pop(video_abs, None)
        return saved

    def _rename_video(self, src: str, dst: str) -> None:
        """Rename a video file and remap its trajectory mapping in lockstep.

        Mirrors the existing ``os.rename(src, dst)`` behaviour but also moves the
        ``video_path -> trajectory_path`` entry so subsequent ``_log_human_preference``
        lookups by video path continue to find the right trajectory ``.pt``.
        """
        src_abs = os.path.abspath(src)
        dst_abs = os.path.abspath(dst)
        if os.path.exists(src):
            os.rename(src, dst)
        if src_abs in self._video_to_traj:
            traj_path = self._video_to_traj.pop(src_abs)
            self._video_to_traj[dst_abs] = traj_path
            meta = self._traj_metadata.get(traj_path)
            if meta is not None:
                meta["role"] = os.path.splitext(os.path.basename(dst))[0]

    def _log_human_preference(
        self,
        *,
        iter_idx: int,
        context: str,
        video_paths: list,
        winner_index: int,
        reward_infos=None,
    ) -> None:
        """Append one record to ``<log_dir>/human_preferences.jsonl``.

        Encodes one entry of υ_h(D_h) (Bobu et al. §4.1): an unordered pair of
        trajectories together with the human's strict preference. Pairs where
        either trajectory is missing (e.g. the side was a "NONE" placeholder) are
        skipped — they cannot be used for σ_TAC because we need both Ĝ_r values.
        """
        if video_paths is None or len(video_paths) != 2:
            return
        v_a, v_b = video_paths[0], video_paths[1]
        if not isinstance(v_a, str) or not isinstance(v_b, str):
            return
        if v_a == "NONE" or v_b == "NONE":
            return

        traj_a = self._video_to_traj.get(os.path.abspath(v_a))
        traj_b = self._video_to_traj.get(os.path.abspath(v_b))

        meta_a = self._traj_metadata.get(traj_a, {}) if traj_a else {}
        meta_b = self._traj_metadata.get(traj_b, {}) if traj_b else {}

        label_a = label_b = None
        if reward_infos is not None and len(reward_infos) >= 2:
            label_a = getattr(reward_infos[0], "name", None)
            label_b = getattr(reward_infos[1], "name", None)

        record = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "iter": int(iter_idx),
            "context": context,
            "winner_index": int(winner_index),
            "video_a": os.path.abspath(v_a),
            "video_b": os.path.abspath(v_b),
            "trajectory_a": traj_a,
            "trajectory_b": traj_b,
            "label_a": label_a,
            "label_b": label_b,
            "iter_role_a": meta_a.get("role"),
            "iter_role_b": meta_b.get("role"),
            "feature_components_a": meta_a.get("feature_components"),
            "feature_components_b": meta_b.get("feature_components"),
            "candidate_reward_function_a": meta_a.get("candidate_reward_function"),
            "candidate_reward_function_b": meta_b.get("candidate_reward_function"),
        }
        if traj_a is None or traj_b is None:
            record["warning"] = "missing_trajectory_skip_in_tac"

        try:
            with open(self._human_pref_jsonl_path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to append to {self._human_pref_jsonl_path}: {exc}")

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
        return total_feed_back_string, success_metric_max, rewards_correlation, oracle_rewards[-1]

    def _log_iteration_results(self, iter: int, results: list):
        """Log the results of the iteration."""
        for idx, result in enumerate(results):
            print(f"{'*' * 20} Iteration {iter} / Process: {idx} {'*' * 20}")
            if result["success"]:
                print(f"Training successful with the following metrics:\n{result['eureka_task_feedback']}")
                print(f"Reward correlation with oracle rewards: {result['reward_correlation']}")
            else:
                print(f"Training failed with the following exception:\n{result['exception']}\n")

        # write the iterations results to file
        with open(f"{self._log_dir}/eureka_iterations.txt", "a") as f:
            for idx, result in enumerate(results):
                f.write(f"{'#' * 20} Iteration: {iter} {'#' * 20}\n\n")
                f.write(f"{'*' * 20} Run: {idx} {'*' * 20}\n")
                feature_components = json.dumps(result['assistant_prompt'], indent=2, default=str)
                f.write(f"- GPT feature components {feature_components}\n")
                f.write(f"- GPT reward method {result['assistant_prompt_rw_gen']}\n")
                f.write(f"- Feature idx: {result['feature_idx']}\n")
                if result["success"]:
                    f.write(f"Training successful with the following metrics:\n{result['eureka_task_feedback']}\n")
                    f.write(f"Reward correlation with oracle rewards:\n{result['reward_correlation']}\n")
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
            output += f"- GPT reward method: {best_run_results['gpt_reward_method']}\n"
            output += f"- Feature components: {best_run_results['feature_components']}\n"
            output += f"- Best training log dir: {best_run_results.get('training_log_dir', 'unknown')}\n"
            output += f"- Best training run dir: {best_run_results.get('training_run_dir', 'unknown')}\n"
            output += f"- Best checkpoint: {best_run_results.get('checkpoint_file', 'unknown')}\n"
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
