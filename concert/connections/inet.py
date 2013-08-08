"""Connection."""
import logbook
import socket


LOG = logbook.Logger(__name__)


class Connection(object):

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
        LOG.debug('Sending {0}'.format(data))
        self._sock.sendall(data.encode('ascii'))

    def recv(self):
        """Read data from the socket."""
        try:
            result = self._sock.recv(1024)
            LOG.debug('Received {0}'.format(result))
            return result
        except socket.timeout:
            LOG.warn('Reading from %s:%i timed out' % self._peer)


class Aerotech(Connection):

    """Aerotech Connection. """
    EOS_CHAR = "\n"  # string termination character
    ACK_CHAR = "%"  # acknowledge
    NAK_CHAR = "!"  # not acknowledge (wrong parameters, etc.)
    FAULT_CHAR = "#"  # task fault

    @classmethod
    def _interpret_response(cls, hle_response):
        if (hle_response[0] == Aerotech.ACK_CHAR):
            # return the data
            res = hle_response[1:
                               hle_response.index(Aerotech.EOS_CHAR)]
            LOG.debug("Interpreted response {0}.".format(res))
            return res
        if (hle_response[0] == Aerotech.NAK_CHAR):
            LOG.warn(hle_response)
            raise ValueError("Invalid command or parameter")
        if (hle_response[0] == Aerotech.FAULT_CHAR):
            raise RuntimeError("Controller task error.")

    def send(self, data):
        """Add eos special character after the command."""
        try:
            super(Aerotech, self).send(data + Aerotech.EOS_CHAR)
        except socket.error as err:
            if err.errno == socket.errno.ECONNRESET:
                LOG.debug("Connection reset by peer, reconnecting...")
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._sock.settimeout(20)
                self._sock.connect(self._peer)
                # Try again.
                super(Aerotech, self).\
                    send(data.upper() + Aerotech.EOS_CHAR)

    def recv(self):
        """Return properly interpreted answer from the controller."""
        return self._interpret_response(super(Aerotech, self).recv())

    def execute(self, data):
        """Execute command and wait for response."""
        self.send(data)
        return self.recv()
