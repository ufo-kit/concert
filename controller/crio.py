import logging
import socket
import threading
import readline

import controller

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


class LinearMotor(controller.LinearMotor):
    def __init__(self, min_limit=-51000, max_limit=51000, port=6342):
        self._connection = _Connection(port)
        self._min_limit = min_limit
        self._max_limit = max_limit

        super(LinearMotor, self).__init__()

    def set_motor_position(self, param, value):
        return self._connection.send('lin %i\r\n' % value)

    def get_limits(self):
        return (self._min_limit, self._max_limit)


class RotationMotor(controller.RotationMotor):
    def __init__(self, port=6340):
        super(RotationMotor, self).__init__()
        self._connection = _Connection(port)

    def set_motor_position(self, param, value):
        return self._connection.send('rot %i\r\n' % value)


if __name__ == '__main__':
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
                rotation_device.rotate(int(value))
            elif command == 'm':
                linear_device.move_absolute(int(value))
        except ValueError:
            print("Commands: `r [NUM]`, `m [NUM]`, `q`")
