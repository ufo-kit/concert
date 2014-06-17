"""New port 74000 monochromator"""

from concert.quantities import q
from concert.networking.base import SocketConnection
from concert.devices.monochromators import base


class NewPort74000(base.Monochromator):

    """
    NewPort 74000 mocochromator class implementation
    """

    def __init__(self, host, port):

        self._connection = SocketConnection(host, port)
        super(NewPort74000, self).__init__()
        self._wavelength = 0 * q.nm

    def shutter_open(self):
        """
        Open the shutter
        """
        self._connection.execute('SHUTTER O\r\n')

    def shutter_close(self):
        """
        Close the shutter
        """
        self._connection.execute('SHUTTER C\r\n')

    def shutter_status(self):
        """
        Get the shutter status
        """
        status = self._connection.execute('SHUTTER?\r\n')[11]
        return status

    def _set_wavelength(self, wave):
        """
        Set the wavelength
        """
        self._connection.execute('GOWAVE ' + str(wave) + '\r\n')

    def _get_wavelength(self):
        """
        Get the current wavelength
        """
        self._wavelength = float(self._connection.execute('WAVE?\r\n')[8:])
        return self._wavelength
        
        
