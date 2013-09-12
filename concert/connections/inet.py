"""Connection."""
import logbook
import socket
from threading import Lock


LOG = logbook.Logger(__name__)


class Connection(object):

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
        self._lock.acquire()
        try:
            self.send(data)
            result = self.recv()
        finally:
            self._lock.release()

        return result


class Aerotech(Connection):

    """Aerotech Connection. """
    EOS_CHAR = "\n"  # string termination character
    ACK_CHAR = "%"  # acknowledge
    NAK_CHAR = "!"  # not acknowledge (wrong parameters, etc.)
    FAULT_CHAR = "#"  # task fault

    def __init__(self, host, port):
        super(Aerotech, self).__init__(host, port,
                                       return_sequence=Aerotech.EOS_CHAR)

    @classmethod
    def _interpret_response(cls, hle_response):
        if (hle_response[0] == Aerotech.ACK_CHAR):
            # return the data
            res = hle_response[1:]
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
