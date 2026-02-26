#!/usr/bin/env bash
#SBATCH --account=rrg-bengioy-ad_gpu
#SBATCH --time=4:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=1
#SBATCH --gpus=h100:1

set -euo pipefail

if command -v module >/dev/null 2>&1; then
    if ! module load StdEnv/2023; then
        echo "[WARNING] Failed to load StdEnv/2023 (possible transient CVMFS issue). Continuing with venv Python."
    fi
fi

VENV_PYTHON="$HOME/llmr/bin/python"
if [ ! -x "$VENV_PYTHON" ]; then
    echo "[ERROR] Python not found in virtualenv: $VENV_PYTHON"
    exit 1
fi

# Ensure local editable package path is importable even if env activation fails.
export PYTHONPATH="$(pwd)/source/isaaclab_eureka${PYTHONPATH:+:$PYTHONPATH}"

export WANDB_MODE=offline

export ISAAC_ACCEPT_EULA=YES

python scripts/train.py --task=Isaac-Quadcopter-Direct-v0 --max_training_iterations=100 --rl_library="rsl_rl"
