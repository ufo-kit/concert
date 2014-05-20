"""EdmundOptics photodiode implementation."""

from concert.quantities import q
from concert.devices.io.wago import IO
from concert.devices.photodiodes import base


class PhotoDiode(base.PhotoDiode):

    """
    Impementation of Edmund Optics photodiode with V output signal
    """

    def __init__(self, host, port, wago_port):
        self._host = host
        self._port = port
        self._wago_port = wago_port
        super(PhotoDiode, self).__init__()

    def _get_intensity(self):
        """
        Read output intensity from the photodiode
        """
        connection = IO(self._host, self._port)
        intensity = float(connection._read_port(self._wago_port)) / 1000
        return intensity * q.V
