import json
import os
import tempfile
import unittest

import numpy as np
import torch

from vlnce_baselines.diagnostic_serialization import json_safe
from vlnce_baselines.evaluation_sensor_diagnostics import write_forensic_layers
from vlnce_baselines.evaluation_telemetry import EvaluationTelemetry
from vlnce_baselines.original_forensics import capture


class DiagnosticSerializationTest(unittest.TestCase):
    def test_json_safe_recursively_converts_numpy_values(self):
        payload = {
            "float32": np.float32(1.25),
            "nested": [np.float64(2.5), {"integer": np.int64(3)}],
            "boolean": np.bool_(True),
            "array": np.array([np.float32(4.5), np.float32(5.5)]),
        }

        converted = json_safe(payload)

        self.assertEqual(
            converted,
            {
                "float32": 1.25,
                "nested": [2.5, {"integer": 3}],
                "boolean": True,
                "array": [4.5, 5.5],
            },
        )
        self.assertIs(converted["float32"].__class__, float)
        self.assertIs(converted["nested"][0].__class__, float)
        self.assertIs(converted["nested"][1]["integer"].__class__, int)
        self.assertIs(converted["boolean"].__class__, bool)
        self.assertEqual(json.loads(json.dumps(converted)), converted)

    def test_three_layer_observation_report_accepts_numpy_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "forensics.json")
            write_forensic_layers(
                path,
                {"rgb": {"mean": np.float32(1.5)}},
                {"depth": {"shape": np.array([2, 3]), "valid": np.bool_(True)}},
            )
            with open(path) as handle:
                report = json.load(handle)

        self.assertEqual(report["sensor_layers"]["raw_habitat"]["rgb"]["mean"], 1.5)
        self.assertEqual(report["sensor_layers"]["batched_pre_policy"]["depth"]["shape"], [2, 3])
        self.assertTrue(report["sensor_layers"]["batched_pre_policy"]["depth"]["valid"])

    def test_summary_report_accepts_numpy_values(self):
        telemetry = EvaluationTelemetry()
        telemetry.start_episode("7", "scene")
        telemetry.finish_episode(
            "7",
            elapsed_ms=np.float64(5.0),
            trajectory=np.array([[np.float32(1.0), np.float32(2.0)]]),
            distances_to_goal=[np.float32(3.0)],
            metrics={"success": np.bool_(True), "steps": np.int64(4)},
        )

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "telemetry.json")
            telemetry.write(path, np.float32(6.0), {"peak": np.int64(7)})
            with open(path) as handle:
                report = json.load(handle)

        self.assertEqual(report["summary"]["evaluation_elapsed_ms"], 6.0)
        self.assertEqual(report["summary"]["cuda_memory"], {"peak": 7})
        self.assertEqual(report["episodes"]["7"]["trajectory"], [[1.0, 2.0]])
        self.assertEqual(report["episodes"]["7"]["metrics"], {"success": True, "steps": 4})

    def test_raw_waypoint_report_accepts_numpy_distance_step(self):
        observations = {
            "rgb": torch.ones(1, 2, 2, 3),
            "depth": torch.ones(1, 2, 2, 1),
        }
        output = {
            "cand_angles": [torch.tensor([0.0])],
            "cand_distances": [torch.tensor([1.0])],
            "cand_img_idxes": [torch.tensor([0])],
        }

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "raw.json")
            capture(
                path,
                observations,
                torch.zeros(1, 120 * 12),
                output,
                distance_step_m=np.float32(0.25),
            )
            with open(path) as handle:
                report = json.load(handle)

        self.assertTrue(
            all(
                entry["distance_m"].__class__ is float
                for entry in report["heatmap"]["top_20_bins"]
            )
        )


if __name__ == "__main__":
    unittest.main()
