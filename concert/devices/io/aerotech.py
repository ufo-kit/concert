"""Aerotech controllers IO boards implementation."""
from concert.networking.aerotech import Connection
from concert.devices.io import base


def _make_ids(index, num_items):
    return zip([index] * num_items, range(num_items))


class HLe(base.IO):

    """Aerotech HLe controller IO board."""

    def __init__(self, host, port=8001):
        super(HLe, self).__init__()
        self._connection = Connection(host, port)
        # Three groups, first 0-5, the rest 0-7
        self._ports = _make_ids(0, 6) + _make_ids(1, 8) + _make_ids(2, 8)

    def _read_port(self, port):
        return int(self._connection.execute("DIN(X,%d,%d)" % (port[0], port[1])))

    def _write_port(self, port, value):
        self._connection.execute("DOUT(X,%d,%d:%d)" % (port[0], port[1], value))
