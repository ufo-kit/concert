"""Dummy photodiode implementation."""
from concert.quantities import q
from concert.devices.photodiodes import base


class PhotoDiode(base.PhotoDiode):

    """A dummy photo diode"""

    def __init__(self):
        super(PhotoDiode, self).__init__()

    def _get_intensity(self):
        return 1 * q.V / q.W
