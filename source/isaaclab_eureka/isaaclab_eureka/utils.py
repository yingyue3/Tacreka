# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from collections import defaultdict

import GPUtil
from isaaclab_eureka.learning_curve_utils import load_tensorboard_scalar_series


def load_tensorboard_logs(path: str):
    """Load tensorboard logs from a given path.

    Args:
        path: The path to the tensorboard logs.

    Returns:
        A dictionary with the tags and their respective values.
    """
    data = defaultdict(list)
    for tag, series in load_tensorboard_scalar_series(path).items():
        data[tag].extend(series["values"])

    return data


def get_freest_gpu():
    """Get the GPU with the most free memory."""
    gpus = GPUtil.getGPUs()
    if not gpus:
        return None
    # Sort GPUs by memory usage
    gpus.sort(key=lambda gpu: gpu.memoryUsed)
    return gpus[0].id


class MuteOutput:
    """Context manager to mute stdout and stderr."""

    def __enter__(self): 
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")  # noqa: SIM115
        sys.stderr = open(os.devnull, "w")  # noqa: SIM115
        return self

    def __exit__(self, *args):
        sys.stdout = self._stdout
        sys.stderr = self._stderr
