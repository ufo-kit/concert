"""Connection to Connection controllers."""
import logging
from concert.networking.base import SocketConnection


LOG = logging.getLogger(__name__)


class Connection(SocketConnection):

    """Aerotech socket connection."""
    EOS_CHAR = "\n"  # string termination character
    ACK_CHAR = "%"  # acknowledge
    NAK_CHAR = "!"  # not acknowledge (wrong parameters, etc.)
    FAULT_CHAR = "#"  # task fault

    def __init__(self, host, port):
        super(Connection, self).__init__(host, port, return_sequence=Connection.EOS_CHAR)

    @classmethod
    def _interpret_response(cls, hle_response):
        if not hle_response:
            raise ValueError("Not enough data received")
        if (hle_response[0] == Connection.ACK_CHAR):
            # return the data
            res = hle_response[1:]
            LOG.debug("Interpreted response {0}.".format(res))
            return res
        if (hle_response[0] == Connection.NAK_CHAR):
            LOG.warn(hle_response)
            raise ValueError("Invalid command or parameter")
        if (hle_response[0] == Connection.FAULT_CHAR):
            raise RuntimeError("Controller task error.")

    def recv(self):
        """Return properly interpreted answer from the controller."""
        return self._interpret_response(super(Connection, self).recv())
