from collections import OrderedDict

import numpy as np


ETP_PANORAMA_HEADINGS_DEGREES = tuple(range(0, 360, 30))


class IsaacPanoramaObservationAdapter(object):
    """Map simultaneous Isaac camera arrays to the legacy ETP panorama.

    This dependency-isolated adapter intentionally accepts NumPy arrays rather
    than Isaac sensor objects.  Runtime acquisition and GPU/CPU transport stay
    outside the ETP package until the pinned Isaac stack can run on a supported
    host.

    Heading zero is robot-forward (+X in the Isaac world/base convention) and
    positive headings rotate counter-clockwise about +Z.  The adapter preserves
    those headings.  The original ETP waypoint provider remains the single
    owner of its historical clockwise packing/reversal.
    """

    def __init__(
        self,
        rgb_height=224,
        rgb_width=224,
        depth_height=256,
        depth_width=256,
        depth_minimum_m=0.0,
        depth_maximum_m=10.0,
    ):
        if rgb_height <= 0 or rgb_width <= 0:
            raise ValueError("RGB dimensions must be positive")
        if depth_height <= 0 or depth_width <= 0:
            raise ValueError("depth dimensions must be positive")
        if depth_maximum_m <= depth_minimum_m:
            raise ValueError("depth range must be strictly increasing")
        self.rgb_shape = (int(rgb_height), int(rgb_width))
        self.depth_shape = (int(depth_height), int(depth_width))
        self.depth_minimum_m = float(depth_minimum_m)
        self.depth_maximum_m = float(depth_maximum_m)

    def adapt(self, rgb_by_heading, depth_by_heading):
        """Return the ordered Habitat-style RGB/depth observation mapping.

        RGB inputs are uint8 HWC/BHWC arrays with three or four channels.
        Depth inputs are metric float HW/HW1/BHW/BHW1 arrays.  Output depth is
        float32 in [0, 1], matching Habitat's normalized 0--10 metre contract.
        No resizing is performed: sensor resolution is part of the experiment.
        """
        self._validate_heading_keys(rgb_by_heading, "RGB")
        self._validate_heading_keys(depth_by_heading, "depth")

        observations = OrderedDict()
        batch_size = None
        for heading in ETP_PANORAMA_HEADINGS_DEGREES:
            rgb = self._prepare_rgb(rgb_by_heading[heading], heading)
            depth = self._prepare_depth(depth_by_heading[heading], heading)
            if batch_size is None:
                batch_size = rgb.shape[0]
            if rgb.shape[0] != batch_size or depth.shape[0] != batch_size:
                raise ValueError(
                    "all panorama views must use one batch size; heading {} "
                    "has RGB/depth batch sizes {}/{} instead of {}".format(
                        heading, rgb.shape[0], depth.shape[0], batch_size
                    )
                )

            suffix = "" if heading == 0 else "_{}".format(heading)
            observations["rgb{}".format(suffix)] = rgb
            observations["depth{}".format(suffix)] = depth
        return observations

    @staticmethod
    def _validate_heading_keys(frames, label):
        expected = set(ETP_PANORAMA_HEADINGS_DEGREES)
        actual = set(frames.keys())
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            raise ValueError(
                "{} panorama headings are invalid: missing={}, extra={}".format(
                    label, missing, extra
                )
            )

    def _prepare_rgb(self, value, heading):
        rgb = np.asarray(value)
        if rgb.dtype != np.uint8:
            raise ValueError(
                "Isaac RGB heading {} must be uint8, received {}".format(
                    heading, rgb.dtype
                )
            )
        if rgb.ndim == 3:
            rgb = rgb[None, ...]
        if rgb.ndim != 4 or rgb.shape[-1] not in (3, 4):
            raise ValueError(
                "Isaac RGB heading {} must have HWC/BHWC shape with 3 or 4 "
                "channels, received {}".format(heading, rgb.shape)
            )
        if rgb.shape[1:3] != self.rgb_shape:
            raise ValueError(
                "Isaac RGB heading {} must be {}x{}, received {}x{}; "
                "configure the sensor instead of silently resizing".format(
                    heading,
                    self.rgb_shape[0],
                    self.rgb_shape[1],
                    rgb.shape[1],
                    rgb.shape[2],
                )
            )
        return np.ascontiguousarray(rgb[..., :3])

    def _prepare_depth(self, value, heading):
        depth = np.asarray(value)
        if depth.ndim == 2:
            depth = depth[None, ..., None]
        elif depth.ndim == 3:
            if depth.shape[-1] == 1:
                depth = depth[None, ...]
            else:
                depth = depth[..., None]
        if depth.ndim != 4 or depth.shape[-1] != 1:
            raise ValueError(
                "Isaac depth heading {} must have HW/HW1/BHW/BHW1 shape, "
                "received {}".format(heading, depth.shape)
            )
        if depth.shape[1:3] != self.depth_shape:
            raise ValueError(
                "Isaac depth heading {} must be {}x{}, received {}x{}; "
                "configure the sensor instead of silently resizing".format(
                    heading,
                    self.depth_shape[0],
                    self.depth_shape[1],
                    depth.shape[1],
                    depth.shape[2],
                )
            )

        depth = depth.astype(np.float32, copy=True)
        depth[~np.isfinite(depth)] = self.depth_maximum_m
        np.clip(
            depth,
            self.depth_minimum_m,
            self.depth_maximum_m,
            out=depth,
        )
        depth -= self.depth_minimum_m
        depth /= self.depth_maximum_m - self.depth_minimum_m
        return np.ascontiguousarray(depth)
