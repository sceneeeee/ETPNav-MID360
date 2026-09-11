import unittest

import numpy as np

from vlnce_baselines.observation_provider.isaac import (
    ETP_PANORAMA_HEADINGS_DEGREES,
    IsaacPanoramaObservationAdapter,
)


class IsaacPanoramaObservationAdapterTest(unittest.TestCase):
    def setUp(self):
        self.adapter = IsaacPanoramaObservationAdapter()

    @staticmethod
    def _rgb_frames(batch_size=1, channels=4):
        return {
            heading: np.full(
                (batch_size, 224, 224, channels),
                heading // 30,
                dtype=np.uint8,
            )
            for heading in ETP_PANORAMA_HEADINGS_DEGREES
        }

    @staticmethod
    def _depth_frames(batch_size=1):
        return {
            heading: np.full(
                (batch_size, 256, 256, 1),
                float(heading) / 100.0,
                dtype=np.float32,
            )
            for heading in ETP_PANORAMA_HEADINGS_DEGREES
        }

    def test_outputs_exact_etp_key_order_shapes_and_dtypes(self):
        observations = self.adapter.adapt(
            self._rgb_frames(), self._depth_frames()
        )
        expected_keys = []
        for heading in ETP_PANORAMA_HEADINGS_DEGREES:
            suffix = "" if heading == 0 else "_{}".format(heading)
            expected_keys.extend(
                ["rgb{}".format(suffix), "depth{}".format(suffix)]
            )

        self.assertEqual(list(observations.keys()), expected_keys)
        self.assertEqual(observations["rgb"].shape, (1, 224, 224, 3))
        self.assertEqual(observations["depth"].shape, (1, 256, 256, 1))
        self.assertEqual(observations["rgb"].dtype, np.uint8)
        self.assertEqual(observations["depth"].dtype, np.float32)

    def test_positive_isaac_yaw_keeps_counter_clockwise_etp_suffix(self):
        observations = self.adapter.adapt(
            self._rgb_frames(), self._depth_frames()
        )
        self.assertTrue(np.all(observations["rgb_90"] == 3))
        self.assertAlmostEqual(float(observations["depth_90"][0, 0, 0, 0]), 0.09)

    def test_metric_depth_is_clipped_normalized_and_nonfinite_is_far(self):
        depths = self._depth_frames()
        depths[0][0, 0, 0, 0] = -1.0
        depths[0][0, 0, 1, 0] = 5.0
        depths[0][0, 0, 2, 0] = 12.0
        depths[0][0, 0, 3, 0] = np.inf
        depths[0][0, 0, 4, 0] = np.nan
        output = self.adapter.adapt(self._rgb_frames(), depths)["depth"]
        self.assertTrue(
            np.allclose(output[0, 0, :5, 0], [0.0, 0.5, 1.0, 1.0, 1.0])
        )

    def test_rejects_missing_heading_and_silent_resize(self):
        rgb = self._rgb_frames()
        del rgb[330]
        with self.assertRaisesRegex(ValueError, "missing=\\[330\\]"):
            self.adapter.adapt(rgb, self._depth_frames())

        rgb = self._rgb_frames()
        rgb[30] = np.zeros((1, 256, 256, 4), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "silently resizing"):
            self.adapter.adapt(rgb, self._depth_frames())


if __name__ == "__main__":
    unittest.main()
