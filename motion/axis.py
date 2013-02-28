import quantities as pq


class LimitReached(Exception):
    pass


def is_out_of_limits(value, limit):
    return limit and value < limit[0] or value > limit[1]


class Positionable(object):
    def __init__(self):
        self.position_limit = None
        self.calibration = None

    def home(self):
        pass

    def set_position(self, position):
        if is_out_of_limits(position, self.position_limit):
            raise LimitReached

        # fire event
        self._set_position_real(position)

    def _set_position_real(self, position):
        raise NotImplementedError

    def get_position(self):
        return self.calibration.to_user(self.feedback.position)


class Velocitable(object):
    def __init__(self):
        self.velocity_limit = None
        self.calibration = None

    def set_velocity(self, velocity):
        # fire event
        if is_out_of_limits(velocity, self.velocity_limit):
            raise LimitReached

        self._set_velocity_real(self, velocity)

    def _set_velocity_real(self, velocity):
        raise NotImplementedError

    def get_velocity(self):
        return self.calibration.to_user(self.feedback.velocity)
