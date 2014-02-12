"""
Motors based on the CompactRIO controller.
"""
from concert.quantities import q
from concert.networking import SocketConnection
from concert.devices.motors import base


DEFAULT_CRIO_HOST = 'cRIO9074-Motion.ipe.kit.edu'
DEFAULT_CRIO_PORT_LINEAR = 6342
DEFAULT_CRIO_PORT_ROTATIONAL = 6340


class LinearMotor(base.LinearMotor):

    """A linear motor that moves in two directions."""

    def __init__(self, host=DEFAULT_CRIO_HOST, port=DEFAULT_CRIO_PORT_LINEAR):
        super(LinearMotor, self).__init__()
        self._connection = SocketConnection(host, port, '\r\n')
        self._steps = 0
        self._to_device_scale = 50000 / q.mm
        self['position'].lower = 0 * q.mm
        self['position'].upper = 2 * q.mm

    def _get_position(self):
        return self._steps / self._to_device_scale

    def _set_position(self, position):
        steps = position * self._to_device_scale
        self._connection.execute('lin {} 5000'.format(int(steps)))

    def _stop(self):
        self._connection.execute('mod stop')


class RotationMotor(base.ContinuousRotationMotor):

    """A rotational motor."""

    def __init__(self, host=DEFAULT_CRIO_HOST, port=DEFAULT_CRIO_PORT_ROTATIONAL):
        super(RotationMotor, self).__init__()
        self._connection = SocketConnection(host, port, '\r\n')
        self._steps = 0
        self._velocity = 0
        self._to_device_scale = 2 / q.deg

    def _in_hard_limit(self):
        return False

    def _in_velocity_hard_limit(self):
        return False

    def _get_position(self):
        return self._steps / self._to_device_scale

    def _set_position(self, position):
        steps = position * self._to_device_scale
        self._connection.execute('rot {} 500'.format(int(steps)))

    def _get_velocity(self):
        return 0 * q.deg / q.s

    def _set_velocity(self, velocity):
        step_velocity = velocity * self._to_device_scale * q.s
        self._connection.execute('mod {}'.format(int(step_velocity)))

    def _stop(self):
        self._connection.execute('mod stop')
