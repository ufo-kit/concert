"""WAGO-I/O-SYSTEM 750"""

from concert.devices.io.base import Device
from concert.networking import WagoConnection


class Wago(Device):

    """Input/output wago system implementation"""

    def __init__(self, host, port):
        self._host = host
        self._port = port
        self._ports = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12}

    def _write_port(self, port, value):
        """Set input value"""
        connection = WagoConnection(self._host, self._port, self._ports[port])
        connection.send(value)

    def _read_port(self, port):
        """Read output value"""
        connection = WagoConnection(self._host, self._port, self._ports[port])
        response = connection.execute()
        return int(response)
