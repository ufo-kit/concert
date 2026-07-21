"""Dummy photodiode implementation."""
from concert.quantities import q
from concert.devices.photodiodes import base


class PhotoDiode(base.PhotoDiode):

    """A dummy photo diode"""

    async def __ainit__(self, **kwargs):
        await super().__ainit__(**kwargs)

    async def _get_intensity(self):
        return 1 * q.V
