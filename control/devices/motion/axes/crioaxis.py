import logging
import readline
import quantities as q
from control.devices.motion.axes.axis import Axis
from control.devices.motion.axes.calibration import LinearCalibration
from control.connections.socketconnection import SocketConnection


CRIO_HOST = 'cRIO9074-Motion.ka.fzk.de'
CRIO_PORT = 6342


class CrioLinearAxis(Axis):
    def __init__(self):
        calibration = LinearCalibration(50000 / q.mm, -1 * q.mm)
        limit = (0 * q.mm, 2 * q.mm)

        super(CrioLinearAxis, self).__init__(None, calibration, limit)

        self._connection = SocketConnection(CRIO_HOST, CRIO_PORT)
        self._register('position',
                       self._get_position,
                       self._set_position,
                       q.m)

    def _stop_real(self):
        pass

    def _get_position(self):
        raise NotImplementedError

    def _set_position(self, steps):
        self._connection.communicate('lin %i\r\n' % steps)


class CrioRotationAxis(Axis):
    def __init__(self):
        calibration = LinearCalibration(50000 / q.mm, -1 * q.mm)
        limit = (0 * q.mm, 2 * q.mm)

        super(CrioRotationAxis, self).__init__(None, calibration, limit)
        self._connection = SocketConnection(CRIO_HOST, CRIO_PORT)
        self._register('position',
                       self._get_position,
                       self._set_position,
                       q.m)

    def _stop_real(self):
        pass

    def _get_position(self):
        raise NotImplementedError

    def _set_position(self, steps):
        self._connection.communicate('rot %i\r\n' % steps)


if __name__ == '__main__':
    logger = logging.getLogger('crio')
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)

    readline.parse_and_bind('tab: complete')

    linear_device = CrioLinearAxis()
    rotation_device = CrioRotationAxis()

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
