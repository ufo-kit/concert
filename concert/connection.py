"""
A collection of connection methods.
"""
import socket
import os
import logbook


log = logbook.Logger(__name__)


class SocketConnection(object):
    """A two-way socket connection."""

    def __init__(self, host, port):
        self._peer = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(20)
        self._sock.connect(self._peer)

    def __del__(self):
        self._sock.close()

    def send(self, data):
        """Send *data* to the peer."""
        log.debug('Sending {0}'.format(data))
        self._sock.sendall(data.encode('ascii'))

    def recv(self):
        try:
            result = self._sock.recv(1024)
            log.debug('Received {0}'.format(result))
            return result
        except socket.timeout:
            log.warn('Reading from %s:%i timed out' % self._peer)


class TangoConnection(object):
    """A connection to a Tango device."""
    def __init__(self, uri, tango_host=None, tango_port=None):
        import PyTango
        # Set the host and port for connecting to the Tango database.
        # TODO: check if there is a way to adjust the host in PyTango.
        if tango_host is not None and tango_port is not None:
            os.environ["TANGO_HOST"] = "%s:%d" % (tango_host, tango_port)

        self._tango_device = PyTango.DeviceProxy(uri)

    def read_value(self, attribute):
        """Read TANGO *attribute* value."""
        return self._tango_device.read_attribute(attribute).value
