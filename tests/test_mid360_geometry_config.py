import copy
import unittest

import yaml

from vlnce_baselines.waypoint_provider.mid360_geometry import (
    MID360GeometryConfig,
    default_mid360_config_path,
)


class MID360GeometryConfigTest(unittest.TestCase):
    def load_values(self):
        with open(default_mid360_config_path(), "r") as handle:
            return yaml.safe_load(handle)

    def test_default_config_is_valid(self):
        config = MID360GeometryConfig(self.load_values())

        self.assertEqual(config.angular_bins, 120)
        self.assertEqual(config.distance_bins, 12)

    def test_sensor_range_must_cover_candidate_footprint_extent(self):
        invalid = copy.deepcopy(self.load_values())
        invalid["maximum_range_m"] = invalid["maximum_distance_m"]

        with self.assertRaisesRegex(ValueError, "footprint"):
            MID360GeometryConfig(invalid)

    def test_sensor_range_accounts_for_obstacle_vertical_extent(self):
        invalid = copy.deepcopy(self.load_values())
        invalid["maximum_range_m"] = (
            invalid["maximum_distance_m"]
            + invalid["robot_radius_m"]
            + invalid["clearance_margin_m"]
        )

        with self.assertRaisesRegex(ValueError, "obstacle-height"):
            MID360GeometryConfig(invalid)


if __name__ == "__main__":
    unittest.main()
