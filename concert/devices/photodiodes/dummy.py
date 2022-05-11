"""Dummy photodiode implementation."""
from concert.quantities import q
from concert.devices.photodiodes import base


class PhotoDiode(base.PhotoDiode):

    """A dummy photo diode"""

    async def __ainit__(self):
        await super(PhotoDiode, self).__ainit__()

    async def _get_intensity(self):
        return 1 * q.V
