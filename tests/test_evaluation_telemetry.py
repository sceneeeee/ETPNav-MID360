import json
import os
import tempfile
import unittest

import torch

from vlnce_baselines.evaluation_telemetry import EvaluationTelemetry


class EvaluationTelemetryTest(unittest.TestCase):
    def test_trace_collection_does_not_change_fixed_policy_output(self):
        """Telemetry receives copied scalars, never the policy tensors."""
        output = {
            "cand_angles": [[0.1, 0.2]],
            "cand_distances": [[1.0, 1.25]],
            "cand_img_idxes": [torch.tensor([2, 3])],
            "policy_logits": torch.tensor([[0.25, 0.75]]),
        }
        expected_logits = output["policy_logits"].clone()
        telemetry = EvaluationTelemetry()
        telemetry.start_episode("fixed", "scene")
        telemetry.record_waypoint_decision(
            "fixed",
            {
                "step_index": 0,
                "candidates": [
                    {"angle": float(angle), "distance": float(distance), "panorama_image_index": int(image)}
                    for angle, distance, image in zip(
                        output["cand_angles"][0], output["cand_distances"][0], output["cand_img_idxes"][0]
                    )
                ],
                "selected_candidate_index": 1,
            },
        )
        torch.testing.assert_close(output["policy_logits"], expected_logits)
        self.assertEqual(output["cand_distances"], [[1.0, 1.25]])
        self.assertEqual(telemetry.episodes["fixed"]["waypoint_trace"][0]["selected_candidate_index"], 1)
        telemetry.record_waypoint_outcome("fixed", collision=True, distance_to_goal=2.0)
        self.assertTrue(telemetry.episodes["fixed"]["waypoint_trace"][0]["collision"])
        self.assertEqual(telemetry.episodes["fixed"]["waypoint_trace"][0]["distance_to_goal"], 2.0)

    def test_report_records_trajectory_latency_metrics_and_cuda_memory(self):
        telemetry = EvaluationTelemetry()
        telemetry.start_episode(
            episode_id="7",
            scene_id="mp3d/example/example.glb",
            trajectory_id=42,
        )
        telemetry.record_decision("7", 12.0)
        telemetry.record_decision("7", 18.0)
        telemetry.finish_episode(
            episode_id="7",
            elapsed_ms=40.0,
            trajectory=[[0.0, 0.0, 0.0], [3.0, 0.0, 4.0]],
            distances_to_goal=[5.0, 0.0],
            metrics={"success": 1.0, "spl": 1.0},
        )

        report = telemetry.report(
            evaluation_elapsed_ms=50.0,
            cuda_memory={"peak_allocated_mib": 1024.0},
        )

        self.assertEqual(report["summary"]["episode_count"], 1)
        self.assertEqual(report["summary"]["decision_count"], 2)
        self.assertEqual(report["summary"]["inference_latency_ms"]["mean"], 15.0)
        self.assertEqual(report["summary"]["inference_latency_ms"]["p50"], 15.0)
        self.assertAlmostEqual(
            report["summary"]["inference_latency_ms"]["p95"], 17.7
        )
        self.assertEqual(
            report["summary"]["cuda_memory"],
            {"peak_allocated_mib": 1024.0},
        )
        self.assertEqual(
            report["episodes"]["7"]["trajectory"],
            [[0.0, 0.0, 0.0], [3.0, 0.0, 4.0]],
        )
        self.assertEqual(report["episodes"]["7"]["metrics"]["success"], 1.0)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "telemetry.json")
            telemetry.write(
                path,
                evaluation_elapsed_ms=50.0,
                cuda_memory={"peak_allocated_mib": 1024.0},
            )
            with open(path, "r") as handle:
                written = json.load(handle)
        self.assertEqual(written, report)


if __name__ == "__main__":
    unittest.main()
