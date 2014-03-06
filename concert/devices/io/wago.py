"""WAGO-I/O-SYSTEM 750"""

from concert.devices.io import base
from concert.networking.wago import Connection


class IO(base.IO):

    """Input/output wago system implementation"""

    def __init__(self, host, port):
        super(IO, self).__init__()
        self._host = host
        self._port = port
        self._ports = range(256)

    def _write_port(self, port, value):
        """Set input value"""
        connection = Connection(self._host, self._port, port)
        connection.send(value)

    def _read_port(self, port):
        """Read output value"""
        connection = Connection(self._host, self._port, port)
        response = connection.execute()
        return int(response)
