"""
Axes based on the CompactRIO controller.
"""
import readline
from concert.quantities import q
from concert.devices.base import LinearCalibration
from concert.devices.motors.base import Motor
from concert.connections.inet import Connection


CRIO_HOST = 'cRIO9074-Motion.ka.fzk.de'
CRIO_PORT = 6342


class LinearMotor(Motor):

    """A linear motor that moves in two directions."""

    def __init__(self):
        calibration = LinearCalibration(50000 / q.mm, -1 * q.mm)
        super(LinearMotor, self).__init__(calibration)

        self._connection = Connection(CRIO_HOST, CRIO_PORT)
        self['position'].limiter = lambda x: x >= 0 * q.mm and x <= 2 * q.mm

        self._home()
        self._steps = 0

    def _get_position(self):
        return self._steps

    def _set_position(self, steps):
        self._connection.send('lin %i\r\n' % steps)
        self._connection.recv()
        self._steps = steps

    def _stop(self):
        pass

    def _home(self):
        self._set_position(0)


class RotationMotor(Motor):

    """A rotational motor."""

    def __init__(self):
        calibration = LinearCalibration(50000 / q.mm, 0 * q.mm)
        super(RotationMotor, self).__init__(calibration)

        self._steps = 0
        self._connection = Connection(CRIO_HOST, CRIO_PORT)
        self['position'].limiter = lambda x: x >= 0 * q.mm and x <= 2 * q.mm

    def _get_position(self):
        return self._steps

    def _set_position(self, steps):
        self._steps = steps
        self._connection.send('rot %i\r\n' % steps)
        self._connection.recv()

    def _stop(self):
        pass


def main():
    """Main definition."""
    readline.parse_and_bind('tab: complete')

    linear_device = LinearMotor()
    rotation_device = RotationMotor()

    while True:
        line = raw_input('> ')
        if line == 'q':
            break

        try:
            command, value = line.split()
            if command == 'r':
                rotation_device.set_position(float(value) * q.mm)
            elif command == 'm':
                linear_device.set_position(float(value) * q.mm)
        except ValueError:
            print("Commands: `r [NUM]', `m [NUM]', `q'")


if __name__ == '__main__':
    main()
