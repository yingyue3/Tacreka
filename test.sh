#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_ROOT="${LOGS_ROOT:-$REPO_ROOT/logs}"
RECORDINGS_DIR="${RECORDINGS_DIR:-$REPO_ROOT/recordings}"
NUM_EPISODES="${NUM_EPISODES:-5}"
RL_LIBRARY="${RL_LIBRARY:-rsl_rl}"
DEVICE="${DEVICE:-cuda}"
BASELINE_FILTER="${BASELINE_FILTER:-}"
HEADLESS_FLAG="${HEADLESS_FLAG:---headless}"
BASELINES="${BASELINES:-tacreka_sr eureka revolve_full}"
read -r -a BASELINE_LIST <<<"$BASELINES"

mkdir -p "$RECORDINGS_DIR"

find_latest_result_file() {
    local task_name="$1"
    local family_name="$2"
    local latest_result=""
    local latest_stamp=""
    local family run_stamp result_file

    while IFS= read -r result_file; do
        family="$(basename "$(dirname "$(dirname "$(dirname "$result_file")")")")"
        if [[ "$family" != "$family_name" ]]; then
            continue
        fi

        run_stamp="$(basename "$(dirname "$result_file")")"
        if [[ -z "$latest_result" || "$run_stamp" > "$latest_stamp" ]]; then
            latest_result="$result_file"
            latest_stamp="$run_stamp"
        fi
    done < <(find "$LOGS_ROOT" -path "*/${task_name}/*/*_final_result.txt" -type f | sort)

    printf '%s\n' "$latest_result"
}

extract_checkpoint_from_result() {
    local result_file="$1"
    local checkpoint_path=""

    checkpoint_path="$(
        grep -E -- '- Best( candidate)? checkpoint:' "$result_file" \
            | tail -n 1 \
            | sed -E 's/^.*- Best( candidate)? checkpoint: //'
    )"

    if [[ -z "$checkpoint_path" ]]; then
        echo "Could not extract checkpoint from '$result_file'." >&2
        exit 1
    fi
    if [[ ! -f "$checkpoint_path" ]]; then
        echo "Checkpoint '$checkpoint_path' from '$result_file' does not exist." >&2
        exit 1
    fi

    printf '%s\n' "$checkpoint_path"
}

run_cartpole_test() {
    local result_file="$1"
    local checkpoint_path="$2"
    local family run_stamp output_file

    family="$(basename "$(dirname "$(dirname "$(dirname "$result_file")")")")"
    run_stamp="$(basename "$(dirname "$result_file")")"
    output_file="$RECORDINGS_DIR/cartpole_${family}_${run_stamp}.mp4"

    echo "[INFO] Cartpole result file: $result_file"
    echo "[INFO] Cartpole checkpoint: $checkpoint_path"
    echo "[INFO] Cartpole output: $output_file"

    ./isaaclab.sh -p scripts/record_cartpole_fallback.py \
        --checkpoint "$checkpoint_path" \
        --task Isaac-Cartpole-Direct-v0 \
        --rl_library "$RL_LIBRARY" \
        $HEADLESS_FLAG \
        --device "$DEVICE" \
        --num_envs 1 \
        --num_episodes "$NUM_EPISODES" \
        --max_frames 0 \
        --output_file "$output_file"
}

run_quadcopter_test() {
    local result_file="$1"
    local checkpoint_path="$2"
    local family run_stamp output_file

    family="$(basename "$(dirname "$(dirname "$(dirname "$result_file")")")")"
    run_stamp="$(basename "$(dirname "$result_file")")"
    output_file="$RECORDINGS_DIR/quadcopter_${family}_${run_stamp}.mp4"

    echo "[INFO] Quadcopter result file: $result_file"
    echo "[INFO] Quadcopter checkpoint: $checkpoint_path"
    echo "[INFO] Quadcopter output: $output_file"

    ./isaaclab.sh -p scripts/record_quadcopter_fallback.py \
        --checkpoint "$checkpoint_path" \
        --task Isaac-Quadcopter-Direct-v0 \
        --rl_library "$RL_LIBRARY" \
        $HEADLESS_FLAG \
        --device "$DEVICE" \
        --num_envs 1 \
        --num_episodes "$NUM_EPISODES" \
        --max_frames 0 \
        --output_file "$output_file"
}

run_latest_for_family() {
    local family="$1"
    local cartpole_result_file=""
    local quadcopter_result_file=""
    local cartpole_checkpoint=""
    local quadcopter_checkpoint=""

    cartpole_result_file="$(find_latest_result_file "Isaac-Cartpole-Direct-v0" "$family")"
    if [[ -n "$cartpole_result_file" ]]; then
        cartpole_checkpoint="$(extract_checkpoint_from_result "$cartpole_result_file")"
        run_cartpole_test "$cartpole_result_file" "$cartpole_checkpoint"
    else
        echo "[WARN] No cartpole final result found for baseline '$family' under '$LOGS_ROOT'."
    fi

    quadcopter_result_file="$(find_latest_result_file "Isaac-Quadcopter-Direct-v0" "$family")"
    if [[ -n "$quadcopter_result_file" ]]; then
        quadcopter_checkpoint="$(extract_checkpoint_from_result "$quadcopter_result_file")"
        run_quadcopter_test "$quadcopter_result_file" "$quadcopter_checkpoint"
    else
        echo "[WARN] No quadcopter final result found for baseline '$family' under '$LOGS_ROOT'."
    fi
}

for baseline in "${BASELINE_LIST[@]}"; do
    if [[ -n "$BASELINE_FILTER" && "$baseline" != "$BASELINE_FILTER" ]]; then
        continue
    fi
    run_latest_for_family "$baseline"
done
