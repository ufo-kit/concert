import socket
import os
import logbook


log = logbook.Logger(__name__)


class Connection(object):
    def __init__(self, uri):
        self._uri = uri

    @property
    def uri(self):
        return self._uri

    def communicate(self, cmd, *args):
        raise NotImplementedError


class SocketConnection(Connection):
    """A two-way socket connection."""

    def __init__(self, host, port):
        super(SocketConnection, self).__init__(str(host) + ":" + str(port))
        self._peer = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(20)
        self._sock.connect(self._peer)

    def __del__(self):
        self._sock.close()

    def communicate(self, cmd, *args):
        """Send *cmd* to the peer.

        :param cmd: command to send
        :param data: data to send

        :return: reponse from the peer
        """
        to_send = cmd % (args)
        log.debug('Sending {0}'.format(to_send))
        self._sock.sendall(to_send.encode('ascii'))

        try:
            result = self._sock.recv(1024)
            log.debug('Received {0}'.format(result))
            return result
        except socket.timeout:
            log.warn('Reading from %s:%i timed out' % self._peer)


class TangoConnection(Connection):
    """A connection to a Tango device."""
    def __init__(self, uri, tango_host=None, tango_port=None):
        super(TangoConnection, self).__init__(uri)

        import PyTango
        # Set the host and port for connecting to the Tango database.
        # TODO: check if there is a way to adjust the host in PyTango.
        if tango_host is not None and tango_port is not None:
            os.environ["TANGO_HOST"] = "%s:%d" % (tango_host, tango_port)

        self._tango_device = PyTango.DeviceProxy(self._uri)

    @property
    def tango_device(self):
        return self._tango_device
