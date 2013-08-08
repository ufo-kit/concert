"""Dummy IO"""
from concert.devices.io import base
from concert.devices.io.base import Port


class IO(base.IO):

    """Dummy I/O device implementation."""

    def __init__(self):
        ports = [Port(0, "busy", self.read_port, None, "Camera busy flag."),
                 Port(1, "exposure", self.read_port,
                      None, "Camera exposure flag."),
                 Port(2, "acq_enable", None, self.write_port,
                      "Acquisition enable flag.")]
        super(IO, self).__init__(ports)

        self._ports = dict((port.port_id, 0)
                           for port in self if isinstance(port, Port))

        self._ports[self["busy"].port_id] = 1

    def read_port(self, port):
        return self._ports[port]

    def write_port(self, port, value):
        self._ports[port] = int(bool(value))
