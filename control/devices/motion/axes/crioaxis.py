import logging
import readline
import quantities as pq
from control.devices.motion.axes.axis import Axis
from control.devices.motion.axes.calibration import LinearCalibration


class CrioLinearAxis(Axis):
    def __init__(self, connection):
        super(CrioLinearAxis, self).__init__(connection,
                            LinearCalibration(50000 / pq.mm, -1 * pq.mm),
                            (0 * pq.mm, 2 * pq.mm))

    def _get_position_real(self):
        raise NotImplementedError

    def _set_position_real(self, value):
        steps = self._calibration.to_steps(value)
        self._connection.communicate('lin %i\r\n' % steps)


class CrioRotationAxis(Axis):
    def __init__(self, connection):
        super(CrioRotationAxis, self).__init__(connection,
                            LinearCalibration(50000 / pq.mm, -1 * pq.mm),
                            (0 * pq.mm, 0 * pq.mm))

    def _get_position_real(self):
        raise NotImplementedError

    def _set_position_real(self, value):
        steps = self._calibration.to_steps(value)
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
                rotation_device.set_position(float(value) * pq.mm)
            elif command == 'm':
                linear_device.set_position(float(value) * pq.mm)
        except ValueError:
            print("Commands: `r [NUM]`, `m [NUM]`, `q`")
