# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

"""Record an Isaac Lab RSL-RL policy to a video file using VideoIsaac.

VideoIsaac starts Isaac Sim internally, so this script requires no manual
AppLauncher setup — just configure the constants below and run.

Usage::

    python record_test.py
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Configuration — edit these to change what gets recorded
# ---------------------------------------------------------------------------

TASK        = "Isaac-Lift-Cube-Franka-v0"
CHECKPOINT  = (
    "/home/kate/Documents/Project_0/Tacreka/logs/eureka"
    "/Isaac-Lift-Cube-Franka-v0/2026-04-19_14-48-08"
    "/rl_runs/rsl_rl_franka_lift_2026-04-19_15-00-40_Run-0/model_99.pt"
)
NUM_CLIPS   = 1
CLIP_LENGTH = 200

# ---------------------------------------------------------------------------
# VideoIsaac must be imported before any Isaac/omni modules so it can launch
# AppLauncher before those modules are initialised.
# ---------------------------------------------------------------------------

from isaaclab_eureka.managers import VideoIsaac  # noqa: E402


def main() -> None:
    recorder = VideoIsaac(task=TASK, device="cuda")
    paths = recorder.record(
            checkpoint=CHECKPOINT,
            output_file="./ratings/bbbbting.mp4",
            num_clips=NUM_CLIPS,
            clip_length=CLIP_LENGTH,
        )
    paths = recorder.record(
            checkpoint=CHECKPOINT,
            output_file="./ratings/ppppppting.mp4",
            num_clips=NUM_CLIPS,
            clip_length=CLIP_LENGTH,
        )
    recorder.close()

    print("\nSaved clips:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
