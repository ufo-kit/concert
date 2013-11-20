"""Dummy IO"""
from concert.base import Parameter
from concert.devices.io import base
from collections import defaultdict


class IO(base.IO):

    """Dummy I/O device implementation."""

    def __init__(self):
        # ports = [Port(0, "busy", self.read_port, None, "Camera busy flag."),
        #          Port(1, "exposure", self.read_port,
        #               None, "Camera exposure flag."),
        #          Port(2, "acq_enable", None, self.write_port,
        #               "Acquisition enable flag.")]
        super(IO, self).__init__()

        self._ports = defaultdict(int)
        self._ports[0] = 1

    def read_port(self, port):
        return self._ports[port]

    def write_port(self, value, port):
        self._ports[port] = int(bool(value))

    busy = Parameter(fget=read_port, data=0)
    exposure = Parameter(fget=read_port, data=1)
    acq_enable = Parameter(fset=write_port, data=2)
