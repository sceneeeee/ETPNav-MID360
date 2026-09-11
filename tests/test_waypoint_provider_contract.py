import unittest

import numpy as np
import torch

from vlnce_baselines.waypoint_provider.types import (
    WAYPOINT_OUTPUT_KEYS,
    WaypointOutput,
    validate_waypoint_output,
)


class WaypointProviderContractTest(unittest.TestCase):
    def make_output(self):
        return WaypointOutput(
            cand_rgb=[torch.zeros(2, 512)],
            cand_depth=[torch.zeros(2, 128)],
            cand_angle_fts=[torch.zeros(2, 4)],
            cand_img_idxes=[np.array([0, 1], dtype=np.int64)],
            cand_angles=[[0.0, 0.5]],
            cand_distances=[[0.25, 0.5]],
            pano_rgb=torch.zeros(1, 12, 512),
            pano_depth=torch.zeros(1, 12, 128),
            pano_angle_fts=torch.zeros(12, 4),
            pano_img_idxes=np.arange(12, dtype=np.int64),
        )

    def test_mapping_preserves_original_ten_field_contract(self):
        output = self.make_output()

        self.assertEqual(tuple(output.keys()), WAYPOINT_OUTPUT_KEYS)
        self.assertIs(validate_waypoint_output(output), output)
        self.assertTrue(torch.equal(output.cand_rgb[0], output["cand_rgb"][0]))

    def test_validation_rejects_a_missing_field(self):
        output = self.make_output()
        del output["cand_distances"]

        with self.assertRaisesRegex(ValueError, "cand_distances"):
            validate_waypoint_output(output)

    def test_validation_rejects_an_extra_field(self):
        output = self.make_output()
        output["debug"] = "not part of the ETP contract"

        with self.assertRaisesRegex(ValueError, "debug"):
            validate_waypoint_output(output)


if __name__ == "__main__":
    unittest.main()
