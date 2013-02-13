import math
import logging

class OutOfRangeError(Exception):
    pass

class RangedParameter(object):
    def __init__(self, start, end, default, callback=None):
        self._start = start
        self._end = end
        self._value = default
        self._callbacks = [callback] if callback else []

    def add_callback(self, callback):
        self._callbacks.append(callback)

    def set_value(self, value):
        if self._start <= value <= self._end:
            self._value = value

            for callback in self._callbacks:
                callback(self, value)
        else:
            raise OutOfRangeError('%f <= %f <= %f violated' % (self._start, value, self._end))

    def get_value(self):
        return self._value

    @property
    def range(self):
        return (self._start, self._end)

    value = property(get_value, set_value)


class Controller(object):
    @property
    def parameters(self):
        return self._params

    def set_motor_position(self, param, value):
        raise NotImplementedError


class LinearMotor(Controller):
    def __init__(self):
        min_limit, max_limit = self.get_limits()
        self.logger = logging.getLogger('ctrl.linear')
        self.logger.propagate = True

        self._params = { 'position': RangedParameter(min_limit, max_limit, 1.0,
                                                     self.set_motor_position) }
        self._position = self._params['position']

    def get_limits(self):
        """Returns a tuple (min, max) denoting hard lower and upper limits of
        the motor device."""
        raise NotImplementedError

    def get_position(self):
        return self._position.value

    def move_relative(self, distance):
        self._position.value += distance

    def move_absolute(self, position):
        self.logger.debug("Move to %f", position)
        self._position.value = position

    def move_to_relative_position(self, position):
        min_limit, max_limit = self.get_limits()
        a = max_limit - min_limit
        b = min_limit
        self.move_absolute(a * position + b)

    position = property(get_position, move_absolute)


class RotationMotor(Controller):
    def __init__(self):
        pass


class PseudoRotationMotor(Controller):
    def __init__(self, param_x, param_y):
        self._params = { 'phi': RangedParameter(0.0, 2*math.pi, 0.0, self._param_changed) }
        self._param_x = param_x
        self._param_y = param_y
        self._phi = self._params['phi']
        self._radius = min(param_x.range[1], param_y.range[1]) / 2.0

    def move_relative(self, phi_distance):
        # Check for negative angles or introduce ClampedRangedParameter
        self._phi.value += phi_distance

    def move_absolute(self, phi):
        self._phi.value = phi

    def _param_changed(self, param, value): 
        x = self._radius * math.cos(value)
        y = self._radius * math.sin(value)
        self._param_x.value = x + self._radius
        self._param_y.value = y + self._radius
