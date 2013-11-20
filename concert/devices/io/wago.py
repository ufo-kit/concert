"""WAGO-I/O-SYSTEM 750"""

from concert.devices.io.base import Device
from concert.networking import WagoConnection


class Wago(Device):

    """Input/output wago system implementation"""

    def __init__(self, host, port, address):

        self._connection = WagoConnection(host, port, address)
        self.address = address

    def write_port(self, value):
        """Set input value"""
        self._connection.send(value)

    def read_port(self):
        """Read output value"""
        response = self._connection.execute()
        return response
