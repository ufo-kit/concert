"""New port 74000 monochromator"""

import time
from concert.quantities import q
from concert.networking.base import SocketConnection
from concert.devices.monochromators import base


class NewPort74000(base.Monochromator):

    """
    NewPort 74000 mocochromator class implementation
    """

    def __init__(self, host, port):

        self._host = host
        self._port = port
        super(NewPort74000, self).__init__()
        self._wavelength = 0 * q.nm

    def shutter_open(self):
        """
        Open the shutter
        """
        self._connection = SocketConnection(self._host, self._port)
        self._connection.send('SHUTTER O\r\n')

    def shutter_close(self):
        """
        Close the shutter
        """
        self._connection = SocketConnection(self._host, self._port)
        self._connection.send('SHUTTER C\r\n')

    def shutter_status(self):
        """
        Get the shutter status
        """
        self._connection = SocketConnection(self._host, self._port)
        self._connection.send('SHUTTER?\r\n')
        time.sleep(0.1)
        status = self._connection.recv()[11]
        return status

    def _set_wavelength(self, wave):
        """
        Set the wavelength
        """
        self._connection = SocketConnection(self._host, self._port)
        self._connection.send('GOWAVE ' + wave + '\r\n')

    def _get_wavelength(self):
        """
        Get the current wavelength
        """
        self._connection = SocketConnection(self._host, self._port)
        self._connection.send('WAVE?\r\n')
        time.sleep(0.1)
        self._wavelength = self._connection.recv()[8:]
        return self._wavelength
