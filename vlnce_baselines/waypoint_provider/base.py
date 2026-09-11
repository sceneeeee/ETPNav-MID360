from abc import ABCMeta, abstractmethod


class WaypointProvider(object, metaclass=ABCMeta):
    """Generate ETP candidates and their aligned panorama features."""

    @abstractmethod
    def predict(self, observations, in_train=False):
        """Return the ten-field ETP waypoint mapping for a sensor batch."""
        raise NotImplementedError
