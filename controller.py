import math

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


class LinearMotor(Controller):
    def __init__(self, device):
        self._params = { 'position': RangedParameter(0.0, 10.0, 1.0, self._param_changed) }
        self._position = self._params['position']
        self._device = device

    def _param_changed(self, param, value):
        # This is obviously a good place to actually set the value on the
        # physical motor. This function will be called on any occasion that the
        # self._params['position'] is changed.
        pass

    def get_position(self):
        return self._position.value

    def move_relative(self, distance):
        self._position.value += distance

    def move_absolute(self, position):
        self._position.value = position

    position = property(get_position, move_absolute)


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


def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step


def meshscan(controllers, step, callback=None):
    def scan(controllers, remaining, step, callback):
        if not remaining:
            if callback:
                callback(controllers)
        else:
            controller = remaining[0]

            for param in controller.parameters.values():
                valid = param.range

                for value in drange(valid[0], valid[1], step):
                    param.value = value
                    scan(controllers, remaining[1:], step, callback)

    scan(controllers, controllers, step, callback)
