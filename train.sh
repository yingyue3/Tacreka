#!/usr/bin/env bash
#SBATCH --account=def-mtaylor3
#SBATCH --time=4:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=1
#SBATCH --gpus=h100:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=somjitnath@gmail.com

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

TASK="${1:-${TASK:-Isaac-Quadcopter-Direct-v0}}"
BASELINE="${2:-${BASELINE:-tacreka_sr}}"
RL_LIBRARY="${RL_LIBRARY:-rsl_rl}"
MAX_TRAINING_ITERATIONS="${MAX_TRAINING_ITERATIONS:-100}"
MAX_EUREKA_ITERATIONS="${MAX_EUREKA_ITERATIONS:-10}"
NUM_REWARD_SEEDS="${NUM_REWARD_SEEDS:-5}"
ENV_SEED="${ENV_SEED:-42}"
NUM_PARALLEL_RUNS="${NUM_PARALLEL_RUNS:-3}"
EXTRA_ARGS=("${@:3}")

REPO_ROOT="$(pwd)"

CMD=(
    "$VENV_PYTHON"
    "$REPO_ROOT/scripts/train.py"
    "--task=${TASK}"
    "--max_training_iterations=${MAX_TRAINING_ITERATIONS}"
    "--rl_library=${RL_LIBRARY}"
    "--baseline=${BASELINE}"
    "--max_eureka_iterations=${MAX_EUREKA_ITERATIONS}"
    "--num_reward_seeds=${NUM_REWARD_SEEDS}"
    "--env_seed=${ENV_SEED}"
    "--num_parallel_runs=${NUM_PARALLEL_RUNS}"
    "${EXTRA_ARGS[@]}"
)

echo "[INFO] Task: ${TASK}"
echo "[INFO] Baseline: ${BASELINE}"
echo "[INFO] RL library: ${RL_LIBRARY}"
echo "[INFO] Max training iterations: ${MAX_TRAINING_ITERATIONS}"
echo "[INFO] Max Eureka/Revolve iterations: ${MAX_EUREKA_ITERATIONS}"
echo "[INFO] Reward seeds per candidate: ${NUM_REWARD_SEEDS}"
echo "[INFO] Environment seed range: ${ENV_SEED}-$((ENV_SEED + NUM_REWARD_SEEDS - 1))"
echo "[INFO] Parallel suggestion runs: ${NUM_PARALLEL_RUNS}"
echo "[INFO] Repo root: ${REPO_ROOT}"
echo "[INFO] Command: ${CMD[*]}"

"${CMD[@]}"
