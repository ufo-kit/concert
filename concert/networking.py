"""Connection protocols for network communication."""
import os
import time
import struct
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
        self._lock.acquire()
        try:
            self.send(data)
            result = self.recv()
        finally:
            self._lock.release()

        return result


class Aerotech(SocketConnection):

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
        if not hle_response:
            raise ValueError("Not enough data received")
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

    def recv(self):
        """Return properly interpreted answer from the controller."""
        return self._interpret_response(super(Aerotech, self).recv())


class WagoConnection(object):

    """Wago connection"""

    def __init__(self, host, port, reg_addr):

        self.host = host
        self.port = port
        self.reg_addr = reg_addr

    def send(self, value):
        """Write input data to the certain register"""

        data = struct.pack('>HHHBBHH', 0, 0, 6, 1, 6, self.reg_addr, value)
        LOG.debug('Sending {0}'.format(data[5:6]))
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket._sock.settimeout(20)
        self.socket.connect((self.host, self.port))
        self.socket.send(data)
        time.sleep(.01)
        self.socket.close()

    def execute(self):
        """Read output data from the certain register"""

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket._sock.settimeout(20)
        self.socket.connect((self.host, self.port))
        data = struct.pack('>HHHBBHH', 0, 0, 6, 1, 3, self.reg_addr, 1)
        self.socket.send(data)
        time.sleep(.01)
        response = struct.unpack('>HHHBBBH', self.socket.recv(11))
        self.socket.close()
        if len(response) != 7:
            raise ValueError("Not enough data received")
        LOG.debug('Received {0}'.format(response[6]))
        return response[6]


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
