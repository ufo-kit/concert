from copy import deepcopy
try:
    import Queue as queue_module
except ImportError:
    import queue as queue_module
import logging
import time
import numpy as np
from concert.quantities import q
from concert.imageprocessing import ramp_filter
from concert.async import threaded
from concert.imageprocessing import flat_correct as make_flat_correct
from .base import coroutine


LOG = logging.getLogger(__name__)


@coroutine
def downsize(consumer, x_slice=None, y_slice=None, z_slice=None):
    """
    downsize(consumer, x_slice=None, y_slice=None, z_slice=None)

    Downsize images in 3D and send them to *consumer*. Every argument
    is either a tuple (start, stop, step). *x_slice* operates on
    image width, *y_slice* on its height and *z_slice* on the incoming
    images, i.e. it creates the third time dimension.

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
    while True:
        image = yield
        if z_start <= i and (not z_stop or i < z_stop):
            if k % z_step == 0:
                consumer.send(image[y_start:y_stop:y_step, x_start:x_stop:x_step])
            k += 1
        i += 1


@coroutine
def queue(consumer):
    """
    queue(consumer)

    Store the incoming data in a queue and dispatch in a separate
    thread which prevents the stalling on the "main" data stream.
    """
    item_queue = queue_module.Queue()

    @threaded
    def serve():
        while True:
            item = item_queue.get()
            consumer.send(item)
            item_queue.task_done()

    serve()

    while True:
        item = yield
        item_queue.put(deepcopy(item))


@coroutine
def average_images(consumer):
    """
    average_images(consumer)

    Average images as they come and send them to *consumer*.
    """
    average = None
    i = 0

    while True:
        data = yield
        if average is None:
            average = np.zeros_like(data, dtype=np.float32)
        average = (average * i + data) / (i + 1)
        consumer.send(average)
        i += 1


@coroutine
def sinograms(num_radiographs, consumer, sinograms_volume=None):
    """
    sinograms(num_radiographs, consumer, sinograms_volume=None)

    Convert *num_radiographs* into sinograms and send them to *consumer*.
    The sinograms are sent every time a new radiograph arrives. If there
    is more than *num_radiographs* radiographs, the sinograms are rewritten
    in a ring-buffer fashion. If *sinograms_volume* is given, it must be a 3D
    array and it is used to store the sinograms.
    """
    i = 0

    def is_compatible(radio_shape, sinos_shape):
        return radio_shape[0] == sinos_shape[0] and \
            radio_shape[1] == sinos_shape[2]

    while True:
        radiograph = yield

        if sinograms_volume is None:
            sinograms_volume = np.zeros((radiograph.shape[0], num_radiographs,
                                        radiograph.shape[1]), dtype=radiograph.dtype)
        if not is_compatible(radiograph.shape, sinograms_volume.shape):
            raise ValueError("Incompatible radiograph shape")

        sinograms_volume[:, i % num_radiographs, :] = radiograph
        consumer.send(sinograms_volume)

        i += 1


@coroutine
def flat_correct(flat, consumer, dark=None):
    """
    flat_correct(flat, consumer, dark=None)

    Flat correcting corounte, which takes a *flat* field, a *dark* field (if
    given), calculates a flat corrected radiograph and forwards it to
    *consumer*.
    """
    while True:
        radio = yield
        consumer.send(make_flat_correct(radio, flat, dark=dark))


@coroutine
def absorptivity(consumer):
    r"""
    absorptivity(consumer)

    Get the absorptivity from a flat corrected stream of images.  The intensity
    after the object is defined as :math:`I = I_0 \cdot e^{-\mu t}` and we
    extract the absorptivity :math:`\mu t` from the stream of flat corrected
    images :math:`I / I_0`.
    """
    while True:
        frame = yield
        consumer.send(-np.log(frame))


@coroutine
def stall(consumer, per_shot=10, flush_at=None):
    """
    stall(consumer, per_shot=10, flush_at=None)

    Send items once enough is collected. Collect *per_shot* items and
    send them to *consumer*. The incoming data might represent a collection
    of some kind. If the last item is supposed to be sent regardless the current
    number of collected items, use *flush_at* by which you specify the collection
    size and every time the current item *counter* % *flush_at* == 0 the item
    is sent.
    """
    i = 1

    while True:
        item = yield
        if i % per_shot == 0:
            consumer.send(item)
        elif flush_at and i % flush_at == 0:
            consumer.send(item)
            i = 0

        i += 1


@coroutine
def backproject(center, consumer):
    """
    backproject(center, consumer)

    Filtered backprojection filter. The filter receives a sinogram,
    filters it and based on *center* of rotation it backprojects it.
    The slice is then sent to *consumer*.
    """
    reco = None
    angle_step = None

    def reconstruct(sinogram, center, angle_step, x_indices, y_indices):
        """Reconstruct the slice by backprojecting all the sinogram rows."""
        width = x_indices.shape[0]
        reco = np.zeros((width, width))

        for i, phi in enumerate(np.arange(sinogram.shape[0]) * angle_step):
            pos = np.sin(phi) * x_indices + np.cos(phi) * y_indices + center
            reco += sinogram[i, pos.astype(np.int)]

        return reco

    def filter_sinogram(sinogram, start, filter_ft, width):
        """High-pass 1D filtering of the sinogram rows."""
        sinogram_ft = np.fft.fft(sinogram[:, start:start + width])
        filtered = sinogram_ft * filter_ft
        return np.fft.ifft(filtered).real.astype(np.float32)

    def get_indices(sinogram, center):
        """
        Get x and y indices which will be the base for rotation. The indices
        are created around the *center* in a symmetrical way, so if there is
        more space on one side it is cut away. This way we can mask out the
        region beyond the inscribed circle of the index arrays. Then we can
        be sure the rotated indices will always fall somewhere in the slice,
        thus we don't need to cut indices which saves time.
        """
        half = min(center, sinogram.shape[1] - center)
        y_indices, x_indices = np.mgrid[-half:half, -half:half]
        mask = np.where(np.sqrt(x_indices ** 2 + y_indices ** 2) >= half)
        x_indices[mask] = 0
        y_indices[mask] = 0

        return y_indices, x_indices

    while True:
        sinogram = yield

        if reco is None:
            if center >= sinogram.shape[1]:
                template = 'Center {} must be less than sinogram width {}'
                raise ValueError(template.format(center, sinogram.shape[1]))
            y_indices, x_indices = get_indices(sinogram, center)
            width = x_indices.shape[0]
            ramp = ramp_filter(width)
            angle_step = np.pi / sinogram.shape[0]
            # We need to store the position where we crop the sinogram, that is
            # based on the old center
            start = center - width / 2
            # Since we crop the indices the new center is always in the middle
            center = width / 2

        filtered = filter_sinogram(sinogram, start, ramp, width)
        reco = reconstruct(filtered, center, angle_step, x_indices, y_indices)
        consumer.send(reco)


class PickSlice(object):

    """Pick a slice from a 3D volume."""

    def __init__(self, index):
        self.index = index

    @coroutine
    def __call__(self, consumer):
        """Pick a slice and send it to *consumer*."""
        while True:
            volume = yield
            consumer.send(volume[self.index])


class Timer(object):

    """Timer object measures execution times of coroutine-based workflows. It measures the time
    from when this object receives data until all the subsequent stages finish, e.g.::

        acquire(timer(process()))

    would measure only the time of *process*, no matter how complicated it is and whether it invokes
    subsequent coroutines. Everything what happens in *process* is taken into account.
    This timer does not treat asynchronous operations in a special way, i.e. if you use it like
    this::

        def long_but_async_operation():
            @async
            def process(data):
                long_op(data)

            while True:
                item = yield
                process(item)

        timer(long_but_async_operation())

    the time you truly measure is only the time to forward the data to *long_but_async_operation*
    and the time to *start* the asynchronous operation (e.g. spawning a thread).

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

    @coroutine
    def __call__(self, consumer):
        """Measures the execution time of *consumer*."""
        while True:
            item = yield
            start = time.time()
            consumer.send(item)
            self.durations.append((time.time() - start) * q.s)
