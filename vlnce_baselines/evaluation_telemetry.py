"""Evaluation-only trajectory, latency, and CUDA-memory reporting."""

import json
import os
from collections import OrderedDict

import numpy as np

from vlnce_baselines.diagnostic_serialization import json_safe


def serialize_waypoint_candidates(output, batch_index):
    """Convert one raw provider output row to the telemetry representation.

    This is deliberately a read-only conversion.  Keeping it outside the
    trainer makes the raw-provider-to-JSON contract directly testable.
    """
    return [
        {
            "angle": float(angle),
            "distance": float(distance),
            "panorama_image_index": int(image_index),
        }
        for angle, distance, image_index in zip(
            output["cand_angles"][batch_index],
            output["cand_distances"][batch_index],
            output["cand_img_idxes"][batch_index],
        )
    ]


class EvaluationTelemetry(object):
    """Collect per-episode evidence without participating in policy decisions."""

    def __init__(self):
        self.episodes = OrderedDict()

    def start_episode(self, episode_id, scene_id, trajectory_id=None):
        episode_key = str(episode_id)
        if episode_key not in self.episodes:
            self.episodes[episode_key] = OrderedDict(
                [
                    ("scene_id", scene_id),
                    ("trajectory_id", trajectory_id),
                    ("decision_latency_ms", []),
                    ("waypoint_trace", []),
                ]
            )

    def record_decision(self, episode_id, elapsed_ms):
        episode_key = str(episode_id)
        if episode_key not in self.episodes:
            raise KeyError("episode telemetry was not started: {}".format(episode_key))
        self.episodes[episode_key]["decision_latency_ms"].append(float(elapsed_ms))

    def record_waypoint_decision(self, episode_id, record):
        """Append JSON-safe data after a policy action is chosen."""
        episode_key = str(episode_id)
        if episode_key not in self.episodes:
            raise KeyError("episode telemetry was not started: {}".format(episode_key))
        self.episodes[episode_key]["waypoint_trace"].append(record)

    def record_waypoint_outcome(self, episode_id, collision, distance_to_goal):
        """Attach simulator outcome to the most recent decision record."""
        episode_key = str(episode_id)
        trace = self.episodes[episode_key]["waypoint_trace"]
        if not trace:
            raise ValueError("waypoint outcome has no preceding decision")
        trace[-1]["collision"] = bool(collision)
        trace[-1]["distance_to_goal"] = float(distance_to_goal)

    def finish_episode(
        self,
        episode_id,
        elapsed_ms,
        trajectory,
        distances_to_goal,
        metrics,
    ):
        episode_key = str(episode_id)
        if episode_key not in self.episodes:
            raise KeyError("episode telemetry was not started: {}".format(episode_key))
        episode = self.episodes[episode_key]
        episode["elapsed_ms"] = float(elapsed_ms)
        episode["trajectory"] = trajectory
        episode["distances_to_goal"] = distances_to_goal
        episode["metrics"] = metrics

    def report(self, evaluation_elapsed_ms, cuda_memory):
        latencies = [
            latency
            for episode in self.episodes.values()
            for latency in episode["decision_latency_ms"]
        ]
        latency_summary = OrderedDict()
        if latencies:
            values = np.asarray(latencies, dtype=np.float64)
            latency_summary.update(
                [
                    ("mean", float(np.mean(values))),
                    ("p50", float(np.percentile(values, 50))),
                    ("p95", float(np.percentile(values, 95))),
                    ("minimum", float(np.min(values))),
                    ("maximum", float(np.max(values))),
                ]
            )
        return OrderedDict(
            [
                (
                    "summary",
                    OrderedDict(
                        [
                            ("episode_count", len(self.episodes)),
                            ("decision_count", len(latencies)),
                            ("evaluation_elapsed_ms", float(evaluation_elapsed_ms)),
                            ("inference_latency_ms", latency_summary),
                            ("cuda_memory", cuda_memory),
                        ]
                    ),
                ),
                ("episodes", self.episodes),
            ]
        )

    def write(self, path, evaluation_elapsed_ms, cuda_memory):
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        report = self.report(evaluation_elapsed_ms, cuda_memory)
        with open(path, "w") as handle:
            json.dump(json_safe(report), handle, indent=2)
            handle.write("\n")
