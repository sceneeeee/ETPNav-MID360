from abc import ABCMeta, abstractmethod


class ObservationProvider(object, metaclass=ABCMeta):
    """Acquire one model-facing observation without exposing simulator APIs."""

    @abstractmethod
    def get_observations(self):
        """Return observations in the stable ETP sensor-key contract."""
        raise NotImplementedError
