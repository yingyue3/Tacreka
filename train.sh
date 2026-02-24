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

# Load API credentials from a user-local secrets file.
SECRETS_FILE="$HOME/.config/secrets/openai.env"
if [ -f "$SECRETS_FILE" ]; then
    . "$SECRETS_FILE"
else
    echo "[WARNING] Secrets file not found: $SECRETS_FILE"
fi

# Fail fast if neither native OpenAI nor Azure OpenAI credentials are configured.
if [ -z "${OPENAI_API_KEY:-}" ] && { [ -z "${AZURE_OPENAI_API_KEY:-}" ] || [ -z "${AZURE_OPENAI_ENDPOINT:-}" ]; }; then
    echo "[ERROR] Missing LLM API credentials."
    echo "[ERROR] Set OPENAI_API_KEY, or set both AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT."
    exit 1
fi

"$VENV_PYTHON" scripts/train.py --task=Isaac-Quadcopter-Direct-v0 --max_training_iterations=100 --rl_library="rsl_rl" --baseline="revolve_full" --max_eureka_iterations=30
# "$VENV_PYTHON" scripts/train.py --task=Isaac-Cartpole-Direct-v0 --max_training_iterations=100 --rl_library=\"rsl_rl\" --baseline=\"revolve_full\"
