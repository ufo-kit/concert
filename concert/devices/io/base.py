"""Port, IO Device."""
from concert.devices.base import Device


class IO(Device):

    """The IO device consists of ports which can be readable, writable or
    both.
    """

    def __init__(self):
        super(IO, self).__init__()

    def read_port(self, port):
        """Read a *port*."""
        raise NotImplementedError

    def write_port(self, port, value):
        """Write a *value* to a *port*."""
        raise NotImplementedError
