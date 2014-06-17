"""Connection protocols for network communication."""
import os
import logging
import socket
from threading import Lock


LOG = logging.getLogger(__name__)


class SocketConnection(object):

    """A two-way socket connection. *return_sequence* is a string appended
    after every command indicating the end of it, the default value
    is a newline (\\n).
    """

    def __init__(self, host, port, return_sequence="\n"):
        self._peer = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(20)
        self._lock = Lock()
        self.return_sequence = return_sequence
        self._sock.connect(self._peer)

    def __del__(self):
        self._sock.close()

    def send(self, data):
        """
        Send *data* to the peer. The return sequence characters
        are appended to the data before it is sent.
        """
        LOG.debug('Sending {0}'.format(data))
        data += self.return_sequence
        self._sock.sendall(data.encode('ascii'))

    def recv(self):
        """
        Read data from the socket. The result is first stripped
        from the trailing return sequence characters and then returned.
        """
        try:
            result = self._sock.recv(1024)
            if result.endswith(self.return_sequence):
                # Strip the command-ending character
                result = result.rstrip(self.return_sequence)
            LOG.debug('Received {0}'.format(result))
            return result
        except socket.timeout:
            LOG.warn('Reading from %s:%i timed out' % self._peer)

    def execute(self, data):
        """Execute command and wait for response (thread safe)."""
        with self._lock:
            self.send(data)
            result = self.recv()

        return result


def get_tango_device(uri, peer=None):
    """
    Get a Tango device by specifying its *uri*. If *peer* is given change the
    tango_host specifying which database to connect to. Format is host:port
    as a string.
    """
    import PyTango
    # TODO: check if there is a way to adjust the host in PyTango.
    if peer is not None:
        os.environ["TANGO_HOST"] = peer

    return PyTango.DeviceProxy(uri)
