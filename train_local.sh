#!/usr/bin/env bash

export WANDB_MODE=offline

export ISAAC_ACCEPT_EULA=YES

TASKS=(
    "Isaac-Quadcopter-Direct-v0"
)

BASELINES=(
    "eureka"
    "tacreka_sr"
    # "revolve_full"
)

# env_id values to use for separate local runs.
ENV_IDS=(
    42
    42
    42
)



total_runs=$((${#TASKS[@]} * ${#BASELINES[@]} * ${#ENV_IDS[@]}))
echo "[INFO] Running ${#TASKS[@]} tasks x ${#BASELINES[@]} baselines x ${#ENV_IDS[@]} env_ids = ${total_runs} total runs"

for task in "${TASKS[@]}"; do
    task_slug="$(slugify_task "$task")"
    for baseline in "${BASELINES[@]}"; do
        for env_id in "${ENV_IDS[@]}"; do
            run_name="${task_slug}_${baseline}_env${env_id}"

            echo "[INFO] Running ${run_name} (log=${log_file})"
            
            ./isaaclab.sh -p  scripts/train.py \
                --task=Isaac-Quadcopter-Direct-v0 \
                --max_training_iterations=500 \
                --baseline=${baseline} \
                --max_eureka_iterations=10 \
                --env_seed=${env_id} \

            
            echo "[INFO] Finished ${run_name}"
        done
    done
done

echo "[INFO] All runs completed."