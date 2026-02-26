#!/bin/sh
#SBATCH --account=def-mtaylor3
#SBATCH --time=3:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=2
#SBATCH --job-name=isaaclab-Isaac-Cartpole

module load StdEnv/2023

source ~/llmr/bin/activate

export WANDB_MODE=offline

export ISAAC_ACCEPT_EULA=YES

python scripts/train.py --task=Isaac-Quadcopter-Direct-v0 --max_training_iterations=100 --rl_library="rsl_rl"
