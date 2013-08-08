"""Port, IO Device."""
from concert.devices.base import Device
from concert.base import Parameter


class Port(Parameter):

    """Port is a :class:`.Parameter` with an id. It represents a low-level
    port capable of sending or receiving electrical signals. *port_id* is the
    identification of the port within a device. When the device wants to
    access this port it will refer to it by the *port_id*."""

    def __init__(self, port_id, name, fget=None, fset=None, doc=None):
        if fget is None:
            new_fget = None
        else:
            new_fget = lambda: fget(port_id)
        if fset is None:
            new_fset = None
        else:
            new_fset = lambda value: fset(port_id, value)
        Parameter.__init__(self, name, fget=new_fget, fset=new_fset, doc=doc)
        self.port_id = port_id


class IO(Device):

    """The IO device consists of ports which can be readable, writable or
    both.
    """

    def __init__(self, ports=None):
        super(IO, self).__init__(ports)

    def read_port(self, port):
        """Read a *port*."""
        raise NotImplementedError

    def write_port(self, port, value):
        """Write a *value* to a *port*."""
        raise NotImplementedError
