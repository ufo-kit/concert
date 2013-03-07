class _Calibration(object):
    def to_user(self, value):
        raise NotImplementedError

    def to_steps(self, value):
        raise NotImplementedError


class LinearCalibration(_Calibration):
    def __init__(self, steps_per_unit, offset):
        self._steps_per_unit = steps_per_unit
        self._offset = offset

    def to_user(self, value_in_steps):
        return value_in_steps/self._steps_per_unit + self._offset

    def to_steps(self, value):
        return (value.rescale(1.0/self._steps_per_unit.units) + self._offset)*\
                                                        self._steps_per_unit