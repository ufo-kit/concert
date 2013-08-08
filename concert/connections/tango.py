"""Tango Device Factory."""
import os
import PyTango


class DeviceFactory(object):

    """Tango device factory based on *host* and *port*."""

    def __init__(self, host, port):
        self._host = host
        self._port = port

    def get_device(self, uri):
        """Get Device."""
        # We need to do this all the time for case it was changed by
        # another factory settings.
        # TODO: check if there is a way to adjust the host in PyTango.
        os.environ["TANGO_HOST"] = "%s:%d" % (self._host, self._port)
        return PyTango.DeviceProxy(uri)


class TopoTomo(DeviceFactory):

    """TopoTomo beam line tango device factory."""

    def __init__(self):
        super(TopoTomo, self).__init__("anka-tango", 10018)
