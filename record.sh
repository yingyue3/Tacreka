#!/bin/sh
#SBATCH --account=def-mtaylor3
#SBATCH --time=3:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=isaaclab-Isaac-Cartpole

module load StdEnv/2023

source ~/llmr/bin/activate

export WANDB_MODE=offline

export ISAAC_ACCEPT_EULA=YES

python ./isaaclab.sh -p scripts/play_with_recording.py \
    --checkpoint /home/yingyue/scratch/logs_save_0106/rl_runs/rsl_rl_eureka/cartpole_direct/2026-01-06_2_tacreka_Run-1/model_99.pt \
    --task Isaac-Cartpole-Direct-v0 \
    --rl_library rsl_rl \
    --headless \
    --record_video \
    --output_dir ./recordings