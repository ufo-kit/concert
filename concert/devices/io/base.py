"""Port, IO Device."""
from concert.devices.base import Device


class IO(Device):

    """The IO device consists of ports which can be readable, writable or
    both.
    """

    def __init__(self):
        super(IO, self).__init__()
        self._ports = []

    @property
    def ports(self):
        """Port IDs used by :meth:`.read_port` and :meth:`.write_port`"""
        return self._ports

    def _check(self, port):
        """Check if *port* is a part of the device."""
        if port not in self._ports:
            raise IODeviceError("Port `{}' not found".format(port))

    def read_port(self, port):
        """Read a *port*."""
        self._check(port)
        return self._read_port(port)

    def write_port(self, port, value):
        """Write a *value* to the *port*."""
        self._check(port)
        self._write_port(port, value)

    def _read_port(self, port):
        """Implementation of reading a *port* from the device."""
        raise NotImplementedError

    def _write_port(self, port, value):
        """Implementation of writing a *value* to a *port* on the device."""
        raise NotImplementedError


class IODeviceError(Exception):

    """Specific exception thrown by IO devices."""

    pass
