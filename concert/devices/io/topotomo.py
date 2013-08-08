"""Tomo Table"""
from concert.devices.io.base import IO, Port
from concert.connections.tango import TopoTomo


class TomoTable(IO):

    """Tomo table's I/O board at the TopoTomo beam line at ANKA."""

    def __init__(self):
        ports = [Port((6, 1), "input", self.read_port, None,
                      "Trigger input to the table."),
                 Port(6, "output", None, self.write_port,
                      "Trigger output from the table.")]
        self._device = TopoTomo().get_device("iss/toto/modbusTomotable")
        super(TomoTable, self).__init__(ports)

    def read_port(self, port):
        return self._device.ReadBits(port)

    def write_port(self, port, value):
        self._device.WriteBits((port, int(bool(value))))
