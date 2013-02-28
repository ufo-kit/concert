class Calibration(object):
    def to_user(self, value):
        pass

    def to_steps(self, value):
        pass


class LinearCalibration(object):
    def __init__(self, steps_per_unit, offset):
        self.steps_per_unit = steps_per_unit
        self.offset = offset

    def to_user(self, value_in_steps):
        return 1. / self.steps_per_unit * value_in_steps + self.offset

    def to_steps(self, value):
        return self.steps_per_unit * (value + self.offset)

