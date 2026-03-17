# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

"""Record Quadcopter policy rollouts without Isaac camera rendering.

This module provides a RecordManagerQuadcopter class that runs policy inference
headlessly and generates a lightweight 2D MP4 visualization from simulation
states and observations. It is intended for clusters where Isaac renderer-based
recording is unavailable.
"""

from __future__ import annotations

import math
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from isaaclab_eureka.utils import get_freest_gpu

from isaaclab_eureka.managers import RecordManagerQuad


recorder = RecordManagerQuad(
    task="Isaac-Quadcopter-Direct-v0",
    num_envs=1,
    device="cuda",
    # output_file="./recordings/quadcopter.mp4",
    max_frames=900,
    num_episodes=1,
)
recorder.record(checkpoint="/home/yingyue/scratch/Tacreka/logs/saved_logs/Tacreka_Quad_3-03/checkpoint/quadcopter_direct/2026-03-03_13-01-55_Run-1/model_99.pt", output_file="./recordings/quadcopter_checkpoint.mp4")
recorder.record(checkpoint="/home/yingyue/scratch/Tacreka/logs/rl_runs/rsl_rl_eureka/quadcopter_direct/2026-03-16_23-16-47_Run-2/model_99.pt", output_file="./recordings/quadcopter_checkpoint.mp4")
