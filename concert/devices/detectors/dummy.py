"""A dummy detector."""
from concert.devices.detectors import base
from concert.devices.cameras.dummy import Camera


class Detector(base.Detector):

    """A dummy detector."""

    async def __ainit__(self, *, camera=None, magnification=None, **kwargs):
        self._camera = await Camera() if camera is None else camera
        self._magnification = 3 if magnification is None else magnification
        await super().__ainit__(**kwargs)

    @property
    def camera(self):
        return self._camera

    @property
    def magnification(self):
        return self._magnification
