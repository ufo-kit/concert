"""PCO cameras implementation."""

import asyncio
import time
from datetime import datetime
import numpy as np
from concert.coroutines.base import background
from concert.devices.cameras import base
from concert.devices.cameras.uca import Camera
from concert.quantities import q


class Pco(Camera):

    """Pco camera implemented by libuca."""

    async def __ainit__(self):
        await super().__ainit__('pco')

    async def stream(self):
        """
        stream()

        Grab frames continuously yield them. This is an async generator.
        """
        try:
            await self.set_acquire_mode(self.uca.enum_values.acquire_mode.AUTO)
            await self.set_storage_mode(self.uca.enum_values.storage_mode.RECORDER)
            await self.set_record_mode(self.uca.enum_values.record_mode.RING_BUFFER)
        except Exception:
            pass

        async for image in super().stream():
            yield image


class PCO4000(Pco):

    """PCO.4000 camera implementation."""

    _ALLOWED_DURING_RECORDING = ['trigger', '_trigger_real', '_last_grab_time',
                                 'grab', '_grab_real', '_record_shape', '_record_dtype',
                                 'uca', 'stop_recording', 'convert', 'state', '_get_state',
                                 '_state_value']

    async def __ainit__(self):
        self._lock_access = False
        self._last_grab_time = None
        await super().__ainit__()

    @background
    async def start_recording(self):
        super().start_recording()
        self._lock_access = True

    @background
    async def stop_recording(self):
        self._lock_access = False
        super().stop_recording()

    async def _grab_real(self, index=None):
        # For the PCO.4000 there must be a delay of at least 1.2 second
        # between consecutive grabs, otherwise it crashes. We provide
        # appropriate timeout here.
        current = time.time()
        if self._last_grab_time and current - self._last_grab_time < 1.2:
            await asyncio.sleep(1.2 - current + self._last_grab_time)
        result = await super()._grab_real(index=index)
        self._last_grab_time = time.time()

        return result

    def __getattribute__(self, name):
        if object.__getattribute__(self, '_lock_access') and name \
           not in PCO4000._ALLOWED_DURING_RECORDING:
            raise AttributeError("{} cannot be accessed during recording".format(name))
        return object.__getattribute__(self, name)


class Dimax(Pco, base.BufferedMixin):

    """A pco.dimax camera implementation."""

    @background
    async def start_recording(self):
        await super().start_recording()
        # By low frame rates the camera returns status that it is already recording, whereas it is
        # not. Waiting 1 frame time seems to help.
        time.sleep((1 / self.frame_rate).to(q.s).magnitude)

    async def _readout_real(self, num_frames=None):
        """Readout *num_frames* frames."""
        recorded_frames = await self.get_recorded_frames()

        if num_frames is None:
            num_frames = recorded_frames.magnitude

        if not 0 < num_frames <= recorded_frames:
            raise base.CameraError("Number of frames {} ".format(num_frames)
                                   + "must be more than zero and less than the recorded "
                                   + "number of frames {}".format(recorded_frames))

        try:
            self.uca.start_readout()

            for i in range(num_frames):
                yield await self.grab()
        finally:
            self.uca.stop_readout()


class Timestamp:

    """Read PCO's binary timestamp. *sequence* is an array of 14 unsigned short pixels."""

    def __init__(self, image):
        """Constructor."""
        if image.dtype != np.uint16:
            raise TypeError('Sequence must have type unsigned short int')
        if len(image.shape) != 2 or image.shape[1] < 14:
            raise ValueError('Image must be a 2D image and must be at least 14 pixels wide.')

        # figures holds the individual figures from BCD-coded numbers, e.g a year composed of 4
        # digits is stored in four separate entries
        figures = []
        for num in image[0, :14]:
            # 16 bits per pixel, 4-bit BCD in the last 8 bits -> 2 decimal digits
            figures.append(num >> 4 & 0xf)
            figures.append(num & 0xf)

        self._number = _concatenate_ints(figures[:8])

        try:
            year = _concatenate_ints(figures[8:12])
            month = _concatenate_ints(figures[12:14])
            day = _concatenate_ints(figures[14:16])
            hour = _concatenate_ints(figures[16:18])
            minute = _concatenate_ints(figures[18:20])
            sec = _concatenate_ints(figures[20:22])
            usec = _concatenate_ints(figures[22:])
            self._time = datetime(year, month, day, hour, minute, sec, usec)
        except ValueError:
            raise TimestampError('No valid timestamp found.')

    @property
    def number(self):
        """Image number in the sequence."""
        return self._number

    @property
    def time(self):
        """Date and time when the image was taken."""
        return self._time

    def __repr__(self):
        return 'Timestamp(number={}, time={})'.format(self.number, self.time)


def _concatenate_ints(sequence):
    """Construct a number from *sequence* of integers, e.g. [1, 2, 3] will be transformed to
    123.
    """
    return int(''.join([str(num) for num in sequence]))


class TimestampError(Exception):
    """Bad timestamp error."""
    pass
