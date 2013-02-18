import math
import logging


class OutOfRangeError(Exception):
    pass


class RangedParameter(object):
    def __init__(self, start, end, default, callback=None):
        self.start = start
        self.end = end
        self.current = default
        self.callbacks = [callback] if callback else []

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def set_value(self, value):
        if self.start <= value <= self.end:
            self.current = value

            for callback in self.callbacks:
                callback(self, value)
        else:
            raise OutOfRangeError('{0} out of range'.format(value))

    def get_value(self):
        return self.current

    @property
    def range(self):
        return (self.start, self.end)

    value = property(get_value, set_value)


class Controller(object):
    def set_motor_position(self, param, value):
        raise NotImplementedError


class LinearMotor(Controller):
    def __init__(self, min_limit, max_limit, default):
        self.min_limit = min_limit
        self.max_limit = max_limit
        self.logger = logging.getLogger('ctrl.linear')
        self.logger.propagate = True

        self.params = {
            'position': RangedParameter(min_limit, max_limit, default,
                                        self.set_motor_position)
        }

        self.position = self.params['position']

    def move_relative(self, distance):
        self.position.value += distance

    def move_absolute(self, position):
        # self.logger.debug("Move to %f", position)
        self.position.value = position

    def move_to_relative_position(self, position):
        a = self.max_limit - self.min_limit
        b = self.min_limit
        self.move_absolute(a * position + b)


class RotationMotor(Controller):
    def __init__(self):
        pass


class PseudoRotationMotor(Controller):
    def __init__(self, param_x, param_y):
        self._params = {
            'phi': RangedParameter(0.0, 2 * math.pi, 0.0, self._param_changed)
        }
        self.param_x = param_x
        self.param_y = param_y
        self.phi = self.params['phi']
        self.radius = min(param_x.range[1], param_y.range[1]) / 2.0

    def move_relative(self, phi_distance):
        # Check for negative angles or introduce ClampedRangedParameter
        self.phi.value += phi_distance

    def move_absolute(self, phi):
        self.phi.value = phi

    def _param_changed(self, param, value):
        x = self.radius * math.cos(value)
        y = self.radius * math.sin(value)
        self.param_x.value = x + self.radius
        self.param_y.value = y + self.radius
