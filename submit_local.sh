#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"
REPO_ROOT="$PWD"

# VENV_PYTHON="${VENV_PYTHON:-$HOME/llmr/bin/python}"
# if [ ! -x "$VENV_PYTHON" ]; then
#     echo "[ERROR] Python not found or not executable: $VENV_PYTHON"
#     exit 1
# fi

TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_ROOT/scripts/train.py}"
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "[ERROR] train.py not found: $TRAIN_SCRIPT"
    exit 1
fi

export WANDB_MODE="${WANDB_MODE:-offline}"
export ISAAC_ACCEPT_EULA="${ISAAC_ACCEPT_EULA:-YES}"

TASKS=(
    "Isaac-Quadcopter-Direct-v0"
)

BASELINES=(
    "eureka"
    # "tacreka_sr"
    # "revolve_full"
)

# env_id values to use for separate local runs.
ENV_IDS=(
    42
    42
    42
)

LOG_DIR="${LOG_DIR:-local_logs}"
MAX_TRAINING_ITERATIONS="${MAX_TRAINING_ITERATIONS:-500}"
MAX_EUREKA_ITERATIONS="${MAX_EUREKA_ITERATIONS:-5}"
RL_LIBRARY="${RL_LIBRARY:-rsl_rl}"
EXTRA_ARGS=("$@")

mkdir -p "$LOG_DIR"

slugify_task() {
    case "$1" in
        Isaac-Cartpole-Direct-v0) echo "cartpole" ;;
        Isaac-Quadcopter-Direct-v0) echo "quadcopter" ;;
        Isaac-Humanoid-Direct-v0) echo "humanoid" ;;
        *) echo "$1" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' ;;
    esac
}

total_runs=$((${#TASKS[@]} * ${#BASELINES[@]} * ${#ENV_IDS[@]}))
echo "[INFO] Running ${#TASKS[@]} tasks x ${#BASELINES[@]} baselines x ${#ENV_IDS[@]} env_ids = ${total_runs} total runs"

for task in "${TASKS[@]}"; do
    task_slug="$(slugify_task "$task")"
    for baseline in "${BASELINES[@]}"; do
        for env_id in "${ENV_IDS[@]}"; do
            run_name="${task_slug}_${baseline}_env${env_id}"
            log_file="${LOG_DIR}/${run_name}.log"

            echo "[INFO] Running ${run_name} (log=${log_file})"
            
            ./isaaclab.sh -p  "$TRAIN_SCRIPT" \
                "--task=${task}" \
                "--max_training_iterations=${MAX_TRAINING_ITERATIONS}" \
                "--rl_library=${RL_LIBRARY}" \
                "--baseline=${baseline}" \
                "--max_eureka_iterations=${MAX_EUREKA_ITERATIONS}" \
                "--env_seed=${env_id}" \
                "${EXTRA_ARGS[@]}" >"$log_file" 2>&1

            echo "[INFO] Finished ${run_name}"
        done
    done
done

echo "[INFO] All runs completed."