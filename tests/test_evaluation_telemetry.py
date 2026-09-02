import json
import os
import tempfile
import unittest

from vlnce_baselines.evaluation_telemetry import EvaluationTelemetry


class EvaluationTelemetryTest(unittest.TestCase):
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
