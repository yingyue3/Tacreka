# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

import csv
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


if "isaaclab_eureka" not in sys.modules:
    package = types.ModuleType("isaaclab_eureka")
    package.__path__ = [str(PACKAGE_DIR)]
    sys.modules["isaaclab_eureka"] = package

if "tensorboard" not in sys.modules:
    tensorboard_module = types.ModuleType("tensorboard")
    backend_module = types.ModuleType("tensorboard.backend")
    event_processing_module = types.ModuleType("tensorboard.backend.event_processing")
    event_accumulator_module = types.ModuleType("tensorboard.backend.event_processing.event_accumulator")
    tensorboard_util_module = types.ModuleType("tensorboard.util")
    tensor_util_module = types.ModuleType("tensorboard.util.tensor_util")

    class _EventAccumulatorStub:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("EventAccumulator should not be used in this unit test.")

    event_accumulator_module.EventAccumulator = _EventAccumulatorStub
    tensor_util_module.make_ndarray = lambda tensor_proto: tensor_proto

    sys.modules["tensorboard"] = tensorboard_module
    sys.modules["tensorboard.backend"] = backend_module
    sys.modules["tensorboard.backend.event_processing"] = event_processing_module
    sys.modules["tensorboard.backend.event_processing.event_accumulator"] = event_accumulator_module
    sys.modules["tensorboard.util"] = tensorboard_util_module
    sys.modules["tensorboard.util.tensor_util"] = tensor_util_module

if "GPUtil" not in sys.modules:
    gputil_module = types.ModuleType("GPUtil")
    gputil_module.getGPUs = lambda: []
    sys.modules["GPUtil"] = gputil_module

if "matplotlib" not in sys.modules:
    matplotlib_module = types.ModuleType("matplotlib")
    pyplot_module = types.ModuleType("matplotlib.pyplot")

    class _AxisStub:
        def plot(self, *args, **kwargs):
            return None

        def fill_between(self, *args, **kwargs):
            return None

        def set_title(self, *args, **kwargs):
            return None

        def set_xlabel(self, *args, **kwargs):
            return None

        def set_ylabel(self, *args, **kwargs):
            return None

        def grid(self, *args, **kwargs):
            return None

        def set_axis_off(self):
            return None

        def text(self, *args, **kwargs):
            return None

        def legend(self, *args, **kwargs):
            return None

    class _FigureStub:
        def savefig(self, path, dpi=None):
            Path(path).write_bytes(b"stub")

    matplotlib_module.use = lambda backend: None
    pyplot_module.subplots = lambda *args, **kwargs: (_FigureStub(), _AxisStub())
    pyplot_module.close = lambda figure: None

    sys.modules["matplotlib"] = matplotlib_module
    sys.modules["matplotlib.pyplot"] = pyplot_module

learning_curve_utils = _load_module("isaaclab_eureka.learning_curve_utils", PACKAGE_DIR / "learning_curve_utils.py")
utils = _load_module("isaaclab_eureka.utils", PACKAGE_DIR / "utils.py")


class TestSeedAggregation(unittest.TestCase):
    """Validate multi-seed aggregation without launching Isaac Lab."""

    def setUp(self):
        self._original_scalar_loader = learning_curve_utils.load_tensorboard_scalar_series
        self._original_log_loader = utils.load_tensorboard_logs

    def tearDown(self):
        learning_curve_utils.load_tensorboard_scalar_series = self._original_scalar_loader
        utils.load_tensorboard_logs = self._original_log_loader

    def test_summarize_tensorboard_candidate_uses_seed_mean(self):
        def fake_load_tensorboard_logs(path: str):
            seed_logs = {
                "seed_a": {
                    "Eureka/eureka_total_rewards": [0.0, 0.0, 1.0, 2.0],
                    "Eureka/oracle_total_rewards": [0.0, 0.0, 1.0, 2.0],
                    "Eureka/success_metric": [0.0, 0.0, 1.05, 1.10],
                },
                "seed_b": {
                    "Eureka/eureka_total_rewards": [0.0, 0.0, 1.0, 3.0],
                    "Eureka/oracle_total_rewards": [0.0, 0.0, 1.0, 3.0],
                    "Eureka/success_metric": [0.0, 0.0, 0.95, 0.90],
                },
            }
            return seed_logs[Path(path).name]

        utils.load_tensorboard_logs = fake_load_tensorboard_logs

        summary = utils.summarize_tensorboard_candidate(
            log_dirs=["seed_a", "seed_b"],
            feedback_subsampling=1,
            success_metric_target=1.0,
        )

        self.assertAlmostEqual(summary["success_metric_mean"], 1.0)
        self.assertAlmostEqual(summary["success_metric_stderr"], 0.10)
        self.assertAlmostEqual(summary["rewards_correlation_mean"], 1.0)
        self.assertIn("task_score", summary["feedback"])
        self.assertIn("±", summary["feedback"])
        self.assertEqual(summary["best_seed_index"], 0)
        self.assertEqual(summary["representative_seed_index"], 0)

    def test_summarize_tensorboard_candidate_tracks_best_seed_separately(self):
        def fake_load_tensorboard_logs(path: str):
            seed_logs = {
                "seed_a": {
                    "Eureka/eureka_total_rewards": [0.0, 0.0, 1.0],
                    "Eureka/oracle_total_rewards": [0.0, 0.0, 1.0],
                    "Eureka/success_metric": [0.0, 0.0, 0.2],
                },
                "seed_b": {
                    "Eureka/eureka_total_rewards": [0.0, 0.0, 1.0],
                    "Eureka/oracle_total_rewards": [0.0, 0.0, 1.0],
                    "Eureka/success_metric": [0.0, 0.0, 0.5],
                },
                "seed_c": {
                    "Eureka/eureka_total_rewards": [0.0, 0.0, 1.0],
                    "Eureka/oracle_total_rewards": [0.0, 0.0, 1.0],
                    "Eureka/success_metric": [0.0, 0.0, 2.0],
                },
            }
            return seed_logs[Path(path).name]

        utils.load_tensorboard_logs = fake_load_tensorboard_logs

        summary = utils.summarize_tensorboard_candidate(
            log_dirs=["seed_a", "seed_b", "seed_c"],
            feedback_subsampling=1,
            success_metric_target=0.0,
        )

        self.assertEqual(summary["best_seed_index"], 0)
        self.assertEqual(summary["representative_seed_index"], 1)

    def test_export_learning_curve_artifacts_adds_stderr_for_multi_seed(self):
        def fake_load_tensorboard_scalar_series(path: str):
            scalar_series = {
                "seed_a": {
                    "Eureka/eureka_total_rewards": {
                        "steps": [1.0, 2.0],
                        "values": [1.0, 3.0],
                        "wall_times": [0.0, 1.0],
                    },
                    "Eureka/success_metric": {
                        "steps": [1.0, 2.0],
                        "values": [0.5, 1.0],
                        "wall_times": [0.0, 1.0],
                    },
                },
                "seed_b": {
                    "Eureka/eureka_total_rewards": {
                        "steps": [1.0, 2.0],
                        "values": [3.0, 5.0],
                        "wall_times": [0.0, 1.0],
                    },
                    "Eureka/success_metric": {
                        "steps": [1.0, 2.0],
                        "values": [0.7, 1.2],
                        "wall_times": [0.0, 1.0],
                    },
                },
            }
            return scalar_series[Path(path).name]

        learning_curve_utils.load_tensorboard_scalar_series = fake_load_tensorboard_scalar_series

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            result = learning_curve_utils.export_learning_curve_artifacts(
                log_dir=["seed_a", "seed_b"],
                output_dir=str(output_dir),
                run_name="aggregate_test",
            )

            self.assertIsNotNone(result)
            self.assertEqual(result["num_runs"], 2)
            self.assertTrue(output_dir.joinpath("learning_curves.png").is_file())

            with output_dir.joinpath("learning_curve_data.csv").open() as csv_file:
                rows = list(csv.DictReader(csv_file))

            self.assertEqual(len(rows), 2)
            self.assertAlmostEqual(float(rows[0]["reward"]), 2.0)
            self.assertAlmostEqual(float(rows[0]["reward_stderr"]), 1.0)
            self.assertEqual(rows[0]["reward_count"], "2")


if __name__ == "__main__":
    unittest.main(verbosity=2, exit=True)
