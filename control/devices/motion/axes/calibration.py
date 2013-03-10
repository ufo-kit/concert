class _Calibration(object):
    def to_user(self, value):
        raise NotImplementedError

    def to_steps(self, value):
        raise NotImplementedError


class LinearCalibration(_Calibration):
    """Represents a linear calibration.

    A linear calibration maps a number of motor steps to a real-world unit
    system.

    """
    def __init__(self, steps_per_unit, offset_in_steps):
        self._steps_per_unit = steps_per_unit
        self._offset = offset_in_steps

    def to_user(self, value_in_steps):
        """Convert value_in_steps to user units"""
        return value_in_steps / self._steps_per_unit + self._offset

    def to_steps(self, value):
        """Convert user unit value to motor steps"""
        return (value - self._offset) * self._steps_per_unit
