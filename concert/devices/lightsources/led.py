"""LEDs"""

from concert.quantities import q
from concert.devices.io.wago import IO
from concert.devices.lightsources import base


class Led(base.LightSource):

    """
    The class for controlling of intensity of LEDs by voltage
    """

    def __init__(self, host, port, wago_port):

        self._host = host
        self._port = port
        self._wago_port = wago_port
        self._intensity = 0 * q.V
        super(Led, self).__init__()

    def _set_intensity(self, value):
        """
        Set the intensity by voltage for the LED
        """
        connection = IO(self._host, self._port)
        connection.write_port(self._wago_port, value.magnitude * 1092)
        self._intensity = value

    def _get_intensity(self):
        """
        Get the voltage from the LED corresponding to current intensity
        """
        return self._intensity
