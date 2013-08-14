"""Calibration maps physical units to a device-recognizable step value."""

from concert.quantities import q


class Calibration(object):

    """Interface to convert between user and device units."""

    def to_user(self, value):
        """Return *value* in user units."""
        raise NotImplementedError

    def to_steps(self, value):
        """Return *value* in device units."""
        raise NotImplementedError


class LinearCalibration(Calibration):

    """A linear calibration maps a number of steps to a real-world unit.

    *steps_per_unit* tells how many steps correspond to some unit,
    *offset_in_steps* by how many steps the device is away from some zero
    point.
    """

    def __init__(self, steps_per_unit, offset_in_steps):
        super(LinearCalibration, self).__init__()
        self._steps_per_unit = steps_per_unit
        self._offset = offset_in_steps

    def to_user(self, value_in_steps):
        return value_in_steps * q.count / self._steps_per_unit - self._offset

    def to_steps(self, value):
        res = (value + self._offset) * self._steps_per_unit
        return res.to_base_units().magnitude
