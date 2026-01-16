# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to test a trained policy's success rate on Isaac-Cartpole-Direct-v0."""

import argparse

from isaaclab_eureka.utils import get_freest_gpu


def main(args_cli):
    """Test the trained policy and calculate success rate."""
    from isaaclab.app import AppLauncher

    # parse args from cmdline
    device = args_cli.device
    task = args_cli.task
    checkpoint = args_cli.checkpoint
    num_episodes = args_cli.num_episodes
    num_envs = args_cli.num_envs

    # parse device
    if device == "cuda":
        device_id = get_freest_gpu()
        device = f"cuda:{device_id}"

    # launch app
    app_launcher = AppLauncher(headless=args_cli.headless, device=device)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import torch
    from isaaclab.envs import DirectRLEnvCfg
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg: DirectRLEnvCfg = parse_env_cfg(task)
    env_cfg.sim.device = device
    env_cfg.scene.num_envs = num_envs if num_envs is not None else env_cfg.scene.num_envs
    env = gym.make(task, cfg=env_cfg)

    # Get max episode length from environment
    max_episode_length = env.unwrapped.max_episode_length
    print(f"Max episode length: {max_episode_length}")

    """Load and run the policy."""
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    if args_cli.rl_library == "rsl_rl":
        from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner

        agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
        agent_cfg.device = device

        env = RslRlVecEnvWrapper(env)
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        ppo_runner.load(checkpoint)
        # obtain the trained policy for inference
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

        # Get actual number of environments
        actual_num_envs = env.unwrapped.num_envs
        
        # reset environment
        obs = env.get_observations()
        
        # Track statistics
        completed_episodes = 0
        successful_episodes = 0
        episode_lengths = []
        
        print(f"Testing policy for {num_episodes} episodes with {actual_num_envs} parallel environments...")
        
        # simulate environment
        while simulation_app.is_running() and completed_episodes < num_episodes:
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                actions = policy(obs)
                # env stepping
                obs, rewards, dones, infos = env.step(actions)
                
                # Check for completed episodes
                if dones.any():
                    # Get episode lengths for completed episodes
                    env_ids = torch.where(dones)[0]
                    for env_id in env_ids:
                        # Get episode length from the environment buffer
                        # The buffer should contain the length of the episode that just completed
                        episode_length = env.unwrapped.episode_length_buf[env_id].item()
                        episode_lengths.append(episode_length)
                        
                        # Check if episode is successful (completed full length)
                        # Success means episode_length / max_episode_length >= (1.0 - tolerance)
                        # For Cartpole, success means the pole stayed upright for the full episode
                        success_threshold = 0.99  # 99% of max episode length
                        is_successful = (episode_length / max_episode_length) >= success_threshold
                        
                        if is_successful:
                            successful_episodes += 1
                        
                        completed_episodes += 1
                        if completed_episodes >= num_episodes:
                            break
                        
                        if completed_episodes % 100 == 0:
                            current_success_rate = successful_episodes / completed_episodes
                            print(f"Completed {completed_episodes}/{num_episodes} episodes. Current success rate: {current_success_rate:.2%}")

    elif args_cli.rl_library == "rl_games":
        from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
        from rl_games.common import env_configurations, vecenv
        from rl_games.common.algo_observer import IsaacAlgoObserver
        from rl_games.torch_runner import Runner

        agent_cfg = load_cfg_from_registry(task, "rl_games_cfg_entry_point")
        # parse checkpoint path
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = checkpoint
        agent_cfg["params"]["config"]["device"] = device
        agent_cfg["params"]["config"]["device_name"] = device
        import math
        clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
        clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
        env = RlGamesVecEnvWrapper(env, device, clip_obs, clip_actions)

        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
        )
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

        # set number of actors into agent config
        agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
        # create runner from rl-games
        runner = Runner(IsaacAlgoObserver())
        runner.load(agent_cfg)

        # obtain the agent from the runner
        from rl_games.common.player import BasePlayer

        agent: BasePlayer = runner.create_player()
        agent.restore(checkpoint)
        agent.reset()

        # reset environment
        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs["obs"]
        # required: enables the flag for batched observations
        _ = agent.get_batch_size(obs, 1)
        # initialize RNN states if used
        if agent.is_rnn:
            agent.init_rnn()
        
        # Get actual number of environments
        actual_num_envs = env.unwrapped.num_envs
        
        # Track statistics
        completed_episodes = 0
        successful_episodes = 0
        episode_lengths = []
        
        print(f"Testing policy for {num_episodes} episodes with {actual_num_envs} parallel environments...")
        
        # simulate environment
        while simulation_app.is_running() and completed_episodes < num_episodes:
            # run everything in inference mode
            with torch.inference_mode():
                # convert obs to agent format
                obs = agent.obs_to_torch(obs)
                # agent stepping
                actions = agent.get_action(obs, is_deterministic=True)
                # env stepping
                obs, rewards, dones, infos = env.step(actions)
                
                if isinstance(obs, dict):
                    obs = obs["obs"]

                # Check for completed episodes
                if dones.any():
                    # Get episode lengths for completed episodes
                    env_ids = torch.where(dones)[0]
                    
                    for env_id in env_ids:
                        # Get episode length from the environment buffer
                        # The buffer should contain the length of the episode that just completed
                        episode_length = env.unwrapped.episode_length_buf[env_id].item()
                        episode_lengths.append(episode_length)
                        
                        # Check if episode is successful (completed full length)
                        success_threshold = 0.99  # 99% of max episode length
                        is_successful = (episode_length / max_episode_length) >= success_threshold
                        
                        if is_successful:
                            successful_episodes += 1
                        
                        completed_episodes += 1
                        if completed_episodes >= num_episodes:
                            break
                        
                        if completed_episodes % 100 == 0:
                            current_success_rate = successful_episodes / completed_episodes
                            print(f"Completed {completed_episodes}/{num_episodes} episodes. Current success rate: {current_success_rate:.2%}")

                # perform operations for terminated episodes
                if len(dones) > 0 and dones.any():
                    # reset rnn state for terminated episodes
                    if agent.is_rnn and agent.states is not None:
                        for s in agent.states:
                            s[:, dones, :] = 0.0

    # Calculate final statistics
    if completed_episodes > 0:
        success_rate = successful_episodes / completed_episodes
        avg_episode_length = sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0
        
        print("\n" + "="*60)
        print("TEST RESULTS")
        print("="*60)
        print(f"Total episodes completed: {completed_episodes}")
        print(f"Successful episodes: {successful_episodes}")
        print(f"Success rate: {success_rate:.2%}")
        print(f"Average episode length: {avg_episode_length:.2f} / {max_episode_length}")
        print(f"Max episode length: {max(episode_lengths) if episode_lengths else 0}")
        print(f"Min episode length: {min(episode_lengths) if episode_lengths else 0}")
        print("="*60)
    else:
        print("No episodes completed during testing.")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained policy's success rate.")
    parser.add_argument("--task", type=str, default="Isaac-Cartpole-Direct-v0", help="Name of the task.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--num_episodes", type=int, default=1000, help="Number of episodes to test.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--device", type=str, default="cuda", help="The device to run testing on.")
    parser.add_argument(
        "--rl_library",
        type=str,
        default="rsl_rl",
        choices=["rsl_rl", "rl_games"],
        help="The RL training library used for training.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Force display off at all times.",
    )
    args_cli = parser.parse_args()

    # Run the main function
    main(args_cli)

