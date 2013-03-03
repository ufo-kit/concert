import logging
import socket
import readline
import quantities as pq
from axis import DiscretelyMovable

HOST = 'cRIO9074-Motion.ka.fzk.de'


class _Connection(object):
    def __init__(self, port):
        self.peer = (HOST, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(20)
        self.sock.connect(self.peer)
        self.logger = logging.getLogger('crio.connection')
        self.logger.propagate = True

    def __del__(self):
        self.sock.close()

    def send(self, data):
        self.logger.debug('Sending {0}'.format(data))
        self.sock.sendall(data.encode('ascii'))

        try:
            result = self.sock.recv(1024)
            self.logger.debug('Received {0}'.format(result))
        except socket.timeout:
            self.logger.warning('Reading from %s:%i timed out' % self.peer)


class CrioLinearAxis(DiscretelyMovable):
    def __init__(self, port=6342):
        super(CrioLinearAxis, self).__init__()

        self.calibration = LinearCalibration(50000 / pq.mm, -1 * pq.mm)
        self.position_limit = (0 * pq.mm, 2 * pq.mm)
        self.connection = _Connection(port)

    def _set_position_real(self, value):
        steps = self.calibration.to_steps(value)
        self.connection.send('lin %i\r\n' % steps)


class CrioRotationAxis(DiscretelyMovable):
    def __init__(self, port=6340):
        super(CrioRotationAxis, self).__init__()

        self.calibration = LinearCalibration(50000 / pq.mm, 0 * pq.mm)
        self.connection = _Connection(port)

    def _set_position_real(self, value):
        steps = self.calibration.to_steps(value)
        return self.connection.send('rot %i\r\n' % steps)


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
