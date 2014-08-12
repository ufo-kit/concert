"""PCO cameras implementation."""

import time
from datetime import datetime
import numpy as np
from concert.devices.cameras import base
from concert.devices.cameras.uca import Camera


class Pco(Camera):

    """Pco camera implemented by libuca."""

    def __init__(self):
        super(Pco, self).__init__('pco')

    def stream(self, consumer):
        """stream frames to the *consumer*."""
        try:
            self.acquire_mode = self.uca.enum_values.acquire_mode.AUTO
            self.storage_mode = self.uca.enum_values.storage_mode.RECORDER
            self.record_mode = self.uca.enum_values.record_mode.RING_BUFFER
        except:
            pass

        return super(Pco, self).stream(consumer)


class PCO4000(Pco):

    """PCO.4000 camera implementation."""

    _ALLOWED_DURING_RECORDING = ['trigger', '_trigger_real', '_last_grab_time',
                                 'grab', '_grab_real', '_record_shape', '_record_dtype',
                                 'uca', 'stop_recording']

    def __init__(self):
        self._lock_access = False
        self._last_grab_time = None
        super(PCO4000, self).__init__()

    def start_recording(self):
        super(PCO4000, self).start_recording()
        self._lock_access = True

    def stop_recording(self):
        self._lock_access = False
        super(PCO4000, self).stop_recording()

    def _grab_real(self):
        # For the PCO.4000 there must be a delay of at least one second
        # between consecutive grabs, otherwise it crashes. We provide
        # appropriate timeout here.
        current = time.time()
        if self._last_grab_time and current - self._last_grab_time < 1:
            time.sleep(1 - current + self._last_grab_time)
        result = super(PCO4000, self)._grab_real()
        self._last_grab_time = time.time()

        return result

    def __getattribute__(self, name):
        if object.__getattribute__(self, '_lock_access') and name \
           not in PCO4000._ALLOWED_DURING_RECORDING:
            raise AttributeError('PCO.4000 is inaccessible during recording')
        return object.__getattribute__(self, name)


class Dimax(Pco, base.BufferedMixin):

    """A pco.dimax camera implementation."""

    def __init__(self):
        super(Dimax, self).__init__()

    def _readout_real(self, num_frames=None):
        """Readout *num_frames* frames."""
        if num_frames is None:
            num_frames = self.recorded_frames.magnitude

        if not 0 < num_frames <= self.recorded_frames:
            raise base.CameraError("Number of frames {} ".format(num_frames) +
                                   "must be more than zero and less than the recorded " +
                                   "number of frames {}".format(self.recorded_frames))

        try:
            self.uca.start_readout()

            for i in xrange(num_frames):
                yield self.grab()
        except base.CameraError:
            raise StopIteration
        finally:
            self.uca.stop_readout()


class Timestamp(object):

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


def _concatenate_ints(sequence):
    """Construct a number from *sequence* of integers, e.g. [1, 2, 3] will be transformed to
    123.
    """
    return int(''.join([str(num) for num in sequence]))


class TimestampError(Exception):
    """Bad timestamp error."""
    pass
