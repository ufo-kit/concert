"""New port 74000 monochromator"""

import time
from concert.quantities import q
from concert.networking import SocketConnection
from concert.devices.monochromators.base import Monochromator
from concert.devices.base import LinearCalibration


class NewPort74000(Monochromator):

    """NewPort 74000 mocochromator class implementation"""

    def __init__(self, host, port):

        self._connection = SocketConnection(host, port)
        calibration = LinearCalibration(q.count / q.nm, 0 * q.nm)
        super(NewPort74000, self).__init__(calibration)
        self.steps = 0

    def shutter_open(self):
        """Open shutter"""
        self._connection.send('SHUTTER O\r\n')

    def shutter_close(self):
        """Close shutter"""
        self._connection.send('SHUTTER C\r\n')

    def shutter_status(self):
        """Get the shutter status"""
        self._connection.send('SHUTTER?\r\n')
        time.sleep(0.1)
        status = self._connection.recv()[11]
        return status

    def _set_wavelength(self, wave):
        """Set the wavelength"""
        self._connection.send('GOWAVE ' + wave + '\r\n')

    def _get_wavelength(self):
        """Get the current wavelength"""
        self._connection.send('WAVE?\r\n')
        time.sleep(0.1)
        self.steps = self._connection.recv()[8:]
        return self.steps
