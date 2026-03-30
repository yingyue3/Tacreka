#!/usr/bin/env bash
# deploy_feedback.sh — Launch the local video feedback website.
#
# Usage:
#   ./deploy_feedback.sh                          # serve all videos in recordings/
#   ./deploy_feedback.sh --video-dir ratings/     # serve a different directory
#   ./deploy_feedback.sh recordings/quad_tac_71.mp4 recordings/quad_tac_85.mp4 \
#       --labels "Reward v1" "Reward v2" \
#       --task "Quadcopter hover task" \
#       --rating
#   ./deploy_feedback.sh --port 8890              # use a different port
#   ./deploy_feedback.sh --wait                   # exit once one response received
#
# Remote access (cluster):
#   Run this script on the cluster node, then on your local machine:
#     ssh -L 8889:localhost:8889 user@<hostname>
#   Open http://localhost:8889 in your browser.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Activate virtual environment ─────────────────────────────────────────────
VENV_PATHS=(
    "$SCRIPT_DIR/.venv"
    "$SCRIPT_DIR/../.venv"
    "$HOME/.venv/isaaclab"
    "$HOME/IsaacLab/.venv"
)
ACTIVATED=false
for venv in "${VENV_PATHS[@]}"; do
    if [[ -f "$venv/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "$venv/bin/activate"
        ACTIVATED=true
        echo "[deploy_feedback] Using venv: $venv"
        break
    fi
done

if [[ "$ACTIVATED" == false ]]; then
    echo "[deploy_feedback] No venv found — using system Python."
fi

# ── Defaults ─────────────────────────────────────────────────────────────────
PORT="${FEEDBACK_PORT:-8889}"
VIDEO_DIR="${FEEDBACK_VIDEO_DIR:-$SCRIPT_DIR/recordings}"
EXTRA_ARGS=()

# Allow --port as the first arg without a value pair when called from env
for arg in "$@"; do
    if [[ "$arg" == "--port" || "$arg" == "-p" ]]; then
        EXTRA_ARGS+=("$arg")
    fi
done

# If no positional video files and no --video-dir flag given, default to recordings/
HAS_VIDEO_DIR=false
HAS_VIDEOS=false
for arg in "$@"; do
    if [[ "$arg" == "--video-dir" || "$arg" == "-d" ]]; then
        HAS_VIDEO_DIR=true
    fi
    if [[ "$arg" == *.mp4 ]]; then
        HAS_VIDEOS=true
    fi
done

# Build argument list
SERVE_ARGS=("$@")
if [[ "$HAS_VIDEO_DIR" == false && "$HAS_VIDEOS" == false ]]; then
    SERVE_ARGS+=("--video-dir" "$VIDEO_DIR")
fi

# Ensure port is set
PORT_IN_ARGS=false
for arg in "${SERVE_ARGS[@]}"; do
    if [[ "$arg" == "--port" || "$arg" == "-p" ]]; then
        PORT_IN_ARGS=true
        break
    fi
done
if [[ "$PORT_IN_ARGS" == false ]]; then
    SERVE_ARGS+=("--port" "$PORT")
fi

# ── Launch ───────────────────────────────────────────────────────────────────
echo ""
echo "Starting feedback server…"
echo "  Script: $SCRIPT_DIR/scripts/serve_feedback.py"
echo "  Args  : ${SERVE_ARGS[*]}"
echo ""

python "$SCRIPT_DIR/scripts/serve_feedback.py" "${SERVE_ARGS[@]}"
