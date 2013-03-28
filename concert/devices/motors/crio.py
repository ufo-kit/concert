"""
Axes based on the CompactRIO controller.
"""
import logging
import readline
import quantities as q
from concert.devices.motors.base import Motor, LinearCalibration
from concert.connection import SocketConnection


CRIO_HOST = 'cRIO9074-Motion.ka.fzk.de'
CRIO_PORT = 6342


class LinearMotor(Motor):
    """A linear motor that moves in two directions."""

    def __init__(self):
        calibration = LinearCalibration(50000 / q.mm, -1 * q.mm)
        super(LinearMotor, self).__init__(calibration)

        self._connection = SocketConnection(CRIO_HOST, CRIO_PORT)
        param = self.get_parameter('position')
        param.limiter = lambda x: x >= 0 * q.mm and x <= 2 * q.mm

        self._set_position(0)
        self._steps = 0

    def _get_position(self):
        return self._steps

    def _set_position(self, steps):
        self._connection.send('lin %i\r\n' % steps)
        self._connection.recv()
        self._steps = steps


class RotationMotor(Motor):
    """A rotational motor."""

    def __init__(self):
        calibration = LinearCalibration(50000 / q.mm, 0 * q.mm)
        super(RotationMotor, self).__init__(calibration)

        self._steps = 0
        self._connection = SocketConnection(CRIO_HOST, CRIO_PORT)
        param = self.get_parameter('position')
        param.limiter = lambda x: x >= -50000 and x <= 50000

    def _get_position(self):
        return self._steps

    def _set_position(self, steps):
        self._steps = steps
        self._connection.send('rot %i\r\n' % steps)
        self._connection.recv()


def main():
    logger = logging.getLogger('crio')
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)

    readline.parse_and_bind('tab: complete')

    linear_device = LinearMotor()
    rotation_device = RotationMotor()

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


if __name__ == '__main__':
    main()
