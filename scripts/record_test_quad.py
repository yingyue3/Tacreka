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
recorder.record(checkpoint="/home/yingyue/scratch/Tacreka/logs/saved_logs/Eureka_Quad_2-24/checkpoint/2026-02-24_03-27-58_Run-0/model_99.pt", output_file="./recordings/quad_eureka_2-24.mp4")
recorder.record(checkpoint="/home/yingyue/scratch/Tacreka/logs/rl_runs/rsl_rl_eureka/quadcopter_direct/2026-03-17_15-01-32_Run-0/model_99.pt", output_file="./recordings/quad_tacreka_3-17.mp4")
