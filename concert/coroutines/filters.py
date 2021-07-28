import logging
import time
import numpy as np
from concert.quantities import q
from concert.imageprocessing import flat_correct as make_flat_correct


LOG = logging.getLogger(__name__)


async def downsize(producer, x_slice=None, y_slice=None, z_slice=None):
    """
    downsize(producer, x_slice=None, y_slice=None, z_slice=None)

    Downsize images in 3D. Every argument is either a tuple (start, stop, step). *x_slice* operates
    on image width, *y_slice* on its height and *z_slice* on the incoming images, i.e. it creates
    the third time dimension.

    Note: the *start* index is included in the data and the *stop* index
    is excluded.
    """
    def check_and_create(sl):
        if not sl:
            sl = (0, None, 1)
        return sl

    x_start, x_stop, x_step = check_and_create(x_slice)
    y_start, y_stop, y_step = check_and_create(y_slice)
    z_start, z_stop, z_step = check_and_create(z_slice)

    i = 0
    k = 0
    async for image in producer:
        if z_start <= i and (not z_stop or i < z_stop):
            if k % z_step == 0:
                yield np.copy(image[y_start:y_stop:y_step, x_start:x_stop:x_step])
            k += 1
        i += 1


async def average_images(producer):
    """
    average_images(producer)

    Average images as they come from *producer*.
    """
    average = None
    i = 0

    async for data in producer:
        if average is None:
            average = np.zeros_like(data, dtype=np.float32)
        average = (average * i + data) / (i + 1)
        yield average
        i += 1


async def flat_correct(flat, producer, dark=None):
    """
    flat_correct(flat, producer, dark=None)

    Flat correcting coroutine which takes a *flat* field and a *dark* field (if given) from
    *producer* and calculates a flat corrected radiograph.
    """
    flat = flat.astype(np.float32)

    async for radio in producer:
        yield make_flat_correct(radio, flat, dark=dark)


async def absorptivity(producer):
    r"""
    absorptivity(producer)

    Get the absorptivity from a flat corrected stream of images.  The intensity
    after the object is defined as :math:`I = I_0 \cdot e^{-\mu t}` and we
    extract the absorptivity :math:`\mu t` from the stream of flat corrected
    images :math:`I / I_0`.
    """
    async for frame in producer:
        yield -np.log(frame)


async def stall(producer, per_shot=10, flush_at=None):
    """
    stall(producer, per_shot=10, flush_at=None)

    Send items once enough is collected from *producer*. Collect *per_shot* items. The incoming data
    might represent a collection of some kind. If the last item is supposed to be sent regardless
    the current number of collected items, use *flush_at* by which you specify the collection size
    and every time the current item *counter* % *flush_at* == 0 the item
    is sent.
    """
    i = 1

    async for item in producer:
        if i % per_shot == 0:
            yield item
        elif flush_at and i % flush_at == 0:
            yield item
            i = 0

        i += 1


class Timer(object):

    """
    Timer object measures execution times of coroutine-based workflows. It measures the time from
    when this object receives data until all the subsequent stages finish.
    """

    def __init__(self):
        self.durations = []

    def reset(self):
        """Reset the timer."""
        self.durations = []

    @property
    def duration(self):
        """All iterations summed up."""
        return sum(self.durations)

    @property
    def mean(self):
        """Mean iteration execution time."""
        return self.duration / len(self.durations)

    async def __call__(self, producer):
        """Measures the execution time of *producer*."""
        async for item in producer:
            start = time.perf_counter()
            yield item
            self.durations.append((time.perf_counter() - start) * q.s)
