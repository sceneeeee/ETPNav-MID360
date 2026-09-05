import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import yaml

from vlnce_baselines.config.default import get_config
from vlnce_baselines.models.graph_utils import GraphMap
from vlnce_baselines.ss_trainer_ETP import RLTrainer
from vlnce_baselines.waypoint_provider.mid360_geometry_v1 import (
    GEOMETRY_V1_MODE,
    MID360GeometryV1Config,
    MID360GeometryV1WaypointProvider,
    default_mid360_v1_config_path,
)
from vlnce_baselines.waypoint_provider.original_etp import EncodedPanorama
from vlnce_baselines.waypoint_provider.types import validate_waypoint_output


def observations(value=0.0, size=5):
    result = {}
    for heading in range(0, 360, 30):
        suffix = "" if heading == 0 else "_{}".format(heading)
        result["depth{}".format(suffix)] = torch.full(
            (1, size, size, 1), value, dtype=torch.float32
        )
    return result


def encoded():
    return EncodedPanorama(
        batch_size=1,
        rgb_embedding=None,
        depth_embedding=None,
        rgb_features=torch.zeros(1, 12, 512),
        depth_features=torch.zeros(1, 12, 128),
    )


class GeometryV1ProviderTest(unittest.TestCase):
    def config(self):
        return MID360GeometryV1Config.from_yaml(default_mid360_v1_config_path())

    def provider(self):
        return MID360GeometryV1WaypointProvider(
            rgb_encoder=None,
            depth_encoder=None,
            rgb_pool=None,
            depth_pool=None,
            pano_angle_features=torch.zeros(12, 4),
            pano_image_indexes=np.arange(12),
            config=self.config(),
        )

    def predict(self, provider, obs):
        with patch(
            "vlnce_baselines.waypoint_provider.mid360_geometry_v1.encode_panorama",
            return_value=encoded(),
        ):
            return provider.predict(obs)

    def test_zero_candidate_provider_output_has_valid_empty_feature_shapes(self):
        provider = self.provider()

        output = self.predict(provider, observations(0.0))

        self.assertIs(validate_waypoint_output(output), output)
        self.assertEqual(output["cand_angles"], [[]])
        self.assertEqual(output["cand_distances"], [[]])
        self.assertEqual(output["cand_img_idxes"][0].shape, (0,))
        self.assertEqual(output["cand_rgb"][0].shape, (0, 512))
        self.assertEqual(output["cand_depth"][0].shape, (0, 128))
        self.assertEqual(output["cand_angle_fts"][0].shape, (0, 4))

    def test_integrated_provider_populates_ray_and_epistemic_diagnostics(self):
        provider = self.provider()

        output = self.predict(provider, observations(0.2, size=9))

        self.assertIs(validate_waypoint_output(output), output)
        debug = provider.last_debug[0]
        self.assertGreater(debug["ray_result"].hit_count, 0)
        self.assertTrue(debug["grid"].free_evidence.any())
        self.assertTrue(debug["grid"].occupied_evidence.any())
        self.assertFalse("pre_nms_score" in debug)
        self.assertLessEqual(len(debug["candidates"]), 5)
        for candidate in debug["candidates"]:
            self.assertTrue(debug["polar_evidence"].safe[
                candidate.angle_index, candidate.distance_index
            ])

    def test_provider_is_deterministic_and_labels_are_diagnostic_only(self):
        provider = self.provider()
        obs = observations(0.2, size=9)
        first = self.predict(provider, obs)
        first_kinds = provider.last_debug[0]["candidate_kinds"]
        second = self.predict(provider, obs)
        self.assertEqual(first["cand_angles"], second["cand_angles"])
        self.assertEqual(first["cand_distances"], second["cand_distances"])
        self.assertEqual(first_kinds, provider.last_debug[0]["candidate_kinds"])
        self.assertNotIn("candidate_kinds", output_keys(first))

    def test_v1_config_matches_v0_physical_values_without_radial_utility(self):
        with open(default_mid360_v1_config_path()) as handle:
            v1 = yaml.safe_load(handle)
        baseline_path = default_mid360_v1_config_path().replace(
            "mid360_waypoint_v1.yaml", "mid360_waypoint.yaml"
        )
        with open(baseline_path) as handle:
            v0 = yaml.safe_load(handle)
        self.assertEqual(v1, v0)
        self.assertNotIn("radial_utility", v1)
        self.assertEqual(self.config().maximum_candidates, 5)

    def test_default_mode_remains_original_and_v1_mode_is_selectable(self):
        default = get_config("run_r2r/iter_train.yaml")
        self.assertEqual(default.MODEL.WAYPOINT_PROVIDER, "original_etp")
        fake_net = SimpleNamespace(
            rgb_encoder=None, depth_encoder=None,
            space_pool_rgb=None, space_pool_depth=None,
            pano_angle_fts=torch.zeros(12, 4),
            pano_img_idxes=np.arange(12),
        )
        trainer = object.__new__(RLTrainer)
        trainer.policy = SimpleNamespace(net=fake_net)
        trainer.config = SimpleNamespace(MODEL=SimpleNamespace(
            WAYPOINT_PROVIDER=GEOMETRY_V1_MODE,
            MID360_V1_CONFIG_PATH=default_mid360_v1_config_path(),
        ))
        selected = trainer._evaluation_waypoint_provider()
        self.assertIsInstance(selected, MID360GeometryV1WaypointProvider)

    def test_training_use_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "evaluation-only"):
            self.provider().predict(observations(), in_train=True)

    def test_zero_candidates_are_accepted_by_graphmap_update(self):
        gmap = GraphMap(
            has_real_pos=False, loc_noise=0.5, merge_ghost=True, ghost_aug=0.0
        )
        cur_vp, candidate_vps, candidate_positions = gmap.identify_node(
            np.zeros(3, dtype=np.float32),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            [], [],
        )
        gmap.update_graph(
            prev_vp=None, step_id=1, cur_vp=cur_vp,
            cur_pos=np.zeros(3, dtype=np.float32),
            cur_embeds=torch.zeros(768),
            cand_vp=candidate_vps, cand_pos=candidate_positions,
            cand_embeds=torch.zeros(0, 768), cand_real_pos=None,
        )
        self.assertEqual(gmap.ghost_pos, {})
        self.assertEqual(gmap.ghost_aug_pos, {})
        self.assertEqual(list(gmap.node_pos), [cur_vp])


def output_keys(output):
    return set(output.keys())


if __name__ == "__main__":
    unittest.main()
