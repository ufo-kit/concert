"""Aerotech"""
from concert.devices.io import base
from concert.devices.io.base import Port


class IO(base.IO):

    """IO class implementation."""

    def __init__(self, connection):
        ports = [Port((1, 0), "read", self.read_port, None, "Read port."),
                 Port((1, 1), "write", None, self.write_port, "Write port.")]
        super(IO, self).__init__(ports)
        self._connection = connection

    def read_port(self, port):
        return self._connection.execute("DIN(X,%d,%d)" % (port[0], port[1]))

    def write_port(self, port, value):
        self._connection.execute("DOUT(X,%d,%d:%d)" %
                                 (port[0], port[1], value))
