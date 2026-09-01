from abc import ABCMeta, abstractmethod


class ObservationProvider(object, metaclass=ABCMeta):
    """Acquire one model-facing observation without exposing simulator APIs."""

    @abstractmethod
    def get_observations(self):
        """Return observations in the stable ETP sensor-key contract."""
        raise NotImplementedError


class RGBPanoramaProvider(object, metaclass=ABCMeta):
    """Acquire the ordered 12-view RGB panorama used for ETP semantics.

    A provider returns heading-keyed, calibrated RGB frames and deliberately
    does not imply a depth or point-cloud source.
    """

    @abstractmethod
    def get_rgb_by_heading(self):
        """Return RGB frames for headings 0, 30, ..., 330 degrees."""
        raise NotImplementedError
