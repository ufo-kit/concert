import logging
import readline
import quantities as q
from concert.devices.axes.axis import Axis, LinearCalibration
from concert.connection import SocketConnection


CRIO_HOST = 'cRIO9074-Motion.ka.fzk.de'
CRIO_PORT = 6342


class LinearAxis(Axis):
    """A linear axis based on the CompactRIO controller."""

    def __init__(self):
        calibration = LinearCalibration(50000 / q.mm, -1 * q.mm)

        super(LinearAxis, self).__init__(calibration)

        self._connection = SocketConnection(CRIO_HOST, CRIO_PORT)
        self._register('position',
                       self._get_position,
                       self._set_position,
                       q.m,
                       lambda x: x >= 0 * q.mm and x <= 2 * q.mm)

    def _get_position(self):
        raise NotImplementedError

    def _set_position(self, steps):
        self._connection.communicate('lin %i\r\n' % steps)


class RotationAxis(Axis):
    """A rotational axis based on the CompactRIO controller."""

    def __init__(self):
        calibration = LinearCalibration(50000 / q.mm, -1 * q.mm)

        super(RotationAxis, self).__init__(calibration)

        self._connection = SocketConnection(CRIO_HOST, CRIO_PORT)
        self._register('position',
                       self._get_position,
                       self._set_position,
                       q.m,
                       lambda x: x >= 0 * q.mm and x <= 2 * q.mm)

    def _get_position(self):
        raise NotImplementedError

    def _set_position(self, steps):
        self._connection.communicate('rot %i\r\n' % steps)


if __name__ == '__main__':
    logger = logging.getLogger('crio')
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)

    readline.parse_and_bind('tab: complete')

    linear_device = LinearAxis()
    rotation_device = RotationAxis()

    try:
        input = raw_input
    except:
        pass

    while True:
        line = input('> ')
        if line == 'q':
            break

        try:
            command, value = line.split()
            if command == 'r':
                rotation_device.set_position(float(value) * q.mm)
            elif command == 'm':
                linear_device.set_position(float(value) * q.mm)
        except ValueError:
            print("Commands: `r [NUM]`, `m [NUM]`, `q`")
