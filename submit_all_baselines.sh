#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"

TASKS=(
    "Isaac-Cartpole-Direct-v0"
    "Isaac-Quadcopter-Direct-v0"
)

BASELINES=(
    "tacreka_sr"
    "eureka"
    "revolve_full"
)

LOG_DIR="${LOG_DIR:-slurm_logs}"
DEFAULT_TIME="${DEFAULT_TIME:-24:00:00}"
REVOLVE_FULL_TIME="${REVOLVE_FULL_TIME:-24:00:00}"
MAX_TRAINING_ITERATIONS="${MAX_TRAINING_ITERATIONS:-500}"
MAX_EUREKA_ITERATIONS="${MAX_EUREKA_ITERATIONS:-5}"
NUM_REWARD_SEEDS="${NUM_REWARD_SEEDS:-5}"
ENV_SEED="${ENV_SEED:-1}"
NUM_PARALLEL_RUNS="${NUM_PARALLEL_RUNS:-3}"

mkdir -p "$LOG_DIR"

slugify_task() {
    case "$1" in
        Isaac-Cartpole-Direct-v0) echo "cartpole" ;;
        Isaac-Quadcopter-Direct-v0) echo "quadcopter" ;;
        *) echo "$1" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' ;;
    esac
}

echo "[INFO] Submitting ${#TASKS[@]} tasks x ${#BASELINES[@]} baselines = $((${#TASKS[@]} * ${#BASELINES[@]})) jobs"
echo "[INFO] Reward seeds per candidate: ${NUM_REWARD_SEEDS}"
echo "[INFO] Base env seed: ${ENV_SEED}"

for task in "${TASKS[@]}"; do
    task_slug="$(slugify_task "$task")"
    for baseline in "${BASELINES[@]}"; do
        job_name="${task_slug}_${baseline}"
        wall_time="$DEFAULT_TIME"
        if [ "$baseline" = "revolve_full" ]; then
            wall_time="$REVOLVE_FULL_TIME"
        fi

        job_id="$(
            sbatch \
                --parsable \
                --job-name="$job_name" \
                --output="$LOG_DIR/${job_name}-%j.out" \
                --chdir="$PWD" \
                --time="$wall_time" \
                --export=ALL,MAX_TRAINING_ITERATIONS="$MAX_TRAINING_ITERATIONS",MAX_EUREKA_ITERATIONS="$MAX_EUREKA_ITERATIONS",NUM_REWARD_SEEDS="$NUM_REWARD_SEEDS",ENV_SEED="$ENV_SEED",NUM_PARALLEL_RUNS="$NUM_PARALLEL_RUNS" \
                "$PWD/train.sh" \
                "$task" \
                "$baseline" \
                "$@"
        )"

        echo "[INFO] Submitted $job_name as job $job_id (time=$wall_time)"
    done
done
