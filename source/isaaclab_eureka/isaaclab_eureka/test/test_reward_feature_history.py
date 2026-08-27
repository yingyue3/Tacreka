import json
import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "reward_feature_history.py"
SPEC = importlib.util.spec_from_file_location("reward_feature_history", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
RewardFeatureHistory = MODULE.RewardFeatureHistory

PROMPT_TEMPLATE = "frequent={FREQUENTLY_CHOSEN_FEATURES}\nrare={RARELY_CHOSEN_FEATURES}\nall={FEATURE_HISTORY_JSON}"


def feature(name, signals, weight=1.0):
    return {
        "feature_name": name,
        "intent": "test",
        "measurable_signals": signals,
        "proxy_metric": "test metric",
        "weight": weight,
        "desired_direction": "increase",
        "typical_failure_mode": "none",
    }


class TestRewardFeatureHistory(unittest.TestCase):
    def test_groups_by_signal_and_records_selected_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            history = RewardFeatureHistory(directory)
            history.add_feature_sets(
                0,
                [
                    [feature("upright", ["projected_gravity_b[:, 2]"])],
                    [
                        feature("renamed_upright", ["  PROJECTED_GRAVITY_B[:, 2]  "]),
                        feature("smooth", ["actions"]),
                    ],
                ],
            )
            history.record_evaluation(0, 1, {0: 0.2, 1: 0.9})

            entries = {entry["signal_key"]: entry for entry in history.as_list()}
            upright = entries["projected_gravity_b[:, 2]"]
            self.assertEqual(upright["generated_count"], 2)
            self.assertNotIn("weight", upright["latest_feature"])
            self.assertTrue(all("weight" not in variant for variant in upright["variants"]))
            self.assertEqual([event["selected"] for event in upright["selection_history"]], [False, True])
            self.assertTrue(entries["actions"]["selection_history"][0]["selected"])
            prompt = history.prompt_section(PROMPT_TEMPLATE)
            self.assertIn("0.9", prompt)
            self.assertIn("frequent=", prompt)
            self.assertEqual(upright["selection_history"][0]["success_metric"], 0.2)

            feature_files = list((Path(directory) / "reward_features").glob("iteration_*/*/feature_*.json"))
            self.assertEqual(len(feature_files), 3)
            with (Path(directory) / "reward_features" / "reward_feature_history.json").open() as stream:
                self.assertEqual(json.load(stream)["grouping"], "measurable_signals")
            history_lines = (Path(directory) / "reward_features" / "reward_feature_history.jsonl").read_text().splitlines()
            self.assertEqual(len(history_lines), 2)
            iteration_file = Path(directory) / "reward_features" / "iteration_000" / "all_features.json"
            iteration_payload = json.loads(iteration_file.read_text())
            self.assertEqual(iteration_payload["selected_feature_set_index"], 1)
            self.assertNotIn("weight", iteration_payload["feature_sets"][0]["features"][0])

    def test_no_success_marks_every_generated_set_not_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            history = RewardFeatureHistory(directory)
            history.add_feature_sets(2, [[feature("position", ["root_pos_w"])]] )
            history.record_evaluation(2, None, {0: None})
            self.assertFalse(history.as_list()[0]["selection_history"][0]["selected"])


if __name__ == "__main__":
    unittest.main()
