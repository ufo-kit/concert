"""Dummy IO"""
from concert.devices.io import base


class IO(base.IO):

    """Dummy I/O device implementation."""

    def __init__(self, port_value=0):
        super(IO, self).__init__()
        self._ports = {0: port_value}

    def _read_port(self, port):
        return self._ports[port]

    def _write_port(self, port, value):
        self._ports[port] = value
