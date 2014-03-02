"""Wago modules connection."""
import logging
import socket
import struct
import time


LOG = logging.getLogger(__name__)


class Connection(object):

    """Wago connection"""

    def __init__(self, host, port, reg_addr):

        self.host = host
        self.port = port
        self.reg_addr = reg_addr

    def send(self, value):
        """Write input data to the certain register"""

        data = struct.pack('>HHHBBHH', 0, 0, 6, 1, 6, self.reg_addr, value)
        LOG.debug('Sending {0}'.format(data[5:6]))
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket._sock.settimeout(20)
        self.socket.connect((self.host, self.port))
        self.socket.send(data)
        time.sleep(.01)
        self.socket.close()

    def execute(self):
        """Read output data from the certain register"""

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket._sock.settimeout(20)
        self.socket.connect((self.host, self.port))
        data = struct.pack('>HHHBBHH', 0, 0, 6, 1, 3, self.reg_addr, 1)
        self.socket.send(data)
        time.sleep(.01)
        response = struct.unpack('>HHHBBBH', self.socket.recv(11))
        self.socket.close()
        if len(response) != 7:
            raise ValueError("Not enough data received")
        LOG.debug('Received {0}'.format(response[6]))
        return response[6]
