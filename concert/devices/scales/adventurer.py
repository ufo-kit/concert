"""Adventurer scales."""

from concert.devices.scales import base
from concert.devices.base import Device
from concert.networking import SocketConnection
from concert.quantities import q
from concert.devices.scales.base import WeightError


class ARRW60(base.TarableScales):

    """The ARRW60 model of Adventurer scales."""
    ERROR = "error"
    OK = "ok"

    def __init__(self, host, port):
        super(ARRW60, self).__init__()
        self._connection = SocketConnection(host, port, "\r\n")
        self._states = self._states.union(set([ARRW60.OK, ARRW60.ERROR]))

    def _execute(self, cmd):
        result = self._connection.execute(cmd)
        if "OK!" not in result:
            raise ValueError("Bad command or value")

    def _tare(self):
        self._execute("T")

    def _get_weight(self):
        """
        Get weight from the scales. The command does not return until
        a balanced position is found, thus it can time out.
        """
        res = self._connection.execute("P")
        if "Err8.4" in res:
            self._set_state(ARRW60.ERROR)
            raise WeightError("More than maximum weight loaded")
        else:
            if self.state == ARRW60.ERROR or self.state == Device.NA:
                # Clear the error from before or set OK for the first time
                self._set_state(ARRW60.OK)
            # The returned string contains units
            res = q.Quantity(res).to(q.g)

        return float(res.magnitude) * q.counts
