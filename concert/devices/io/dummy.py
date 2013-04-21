from concert.devices.io import base
from concert.devices.io.base import Port


class IO(base.IO):
    """Dummy I/O device implementation."""
    def __init__(self):
        ports = [Port(0, "busy", self._read_port, None, "Camera busy flag."),
                 Port(1, "exposure", self._read_port,
                      None, "Camera exposure flag."),
                 Port(2, "acq_enable", None, self._write_port,
                      "Acquisition enable flag.")]
        super(IO, self).__init__(ports)
        self._ports = {port.port_id: 0
                       for port in filter(lambda x: x.__class__ == Port, self)}

        self._ports[self["busy"].port_id] = 1

    def _read_port(self, port):
        return self._ports[port]

    def _write_port(self, port, value):
        self._ports[port] = int(bool(value))
