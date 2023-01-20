"""PCO cameras implementation."""

from datetime import datetime
import numpy as np
import logging

from concert.coroutines.base import background

from concert.devices.cameras.uca import Camera as UcaCamera
from concert.helpers import ImageWithMetadata

LOG = logging.getLogger(__name__)


class Camera(UcaCamera):
    async def __ainit__(self, name="pco", params=None):
        self._timestamp_enabled = False
        await super().__ainit__(name=name, params=params)

    async def _record_real(self):
        self._timestamp_enabled = await self.get_timestamp() in ['both', 'binary']
        await super()._record_real()

    @background
    async def start_readout(self):
        self._timestamp_enabled = await self.get_timestamp() in ['both', 'binary']
        await super().start_readout()

    @background
    async def grab(self) -> ImageWithMetadata:
        """Return a concert.storage.ImageWithMetadata (subclass of np.ndarray) with data of the
        current frame.

        If timestamps are enabled, the frame number and the time is added as 'frame_number' and
        'timestamp' to the images metadata dictionary.
        If the timestamp can not be extracted (and it should be there), a TimestampError will be
        raised.
        """
        img = await self._grab_real()
        if self._timestamp_enabled:
            try:
                timestamp = Timestamp(img)
            except TimestampError as e:
                LOG.error("Can not extract timestamp from frame.")
                raise e
            img = self.convert(img)
            img = img.view(ImageWithMetadata)
            img.metadata['frame_number'] = timestamp.number
            img.metadata['timestamp'] = timestamp.time.isoformat()
        return img


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
