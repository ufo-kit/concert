from copy import deepcopy
try:
    import Queue as queue
except ImportError:
    import queue
import logging
import numpy as np
from concert.imageprocessing import get_ramp_filter, backproject as backproject_algorithm
from concert.async import threaded
from concert.coroutines import coroutine
from concert.imageprocessing import flat_correct as make_flat_correct


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
def cache(consumer):
    """
    cache(consumer)

    Cache the incoming data into a queue and dispatch in a separate
    thread which prevents the stalling on the "main" data stream.
    """
    item_queue = queue.Queue()

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
def average_images(num_images, consumer):
    """
    average_images(num_images, consumer)

    Average *num_images* images as they come and send them to *consumer*.
    """
    average = None
    i = 0

    while True:
        data = yield
        if average is None:
            average = np.zeros_like(data, dtype=np.float32)
        average = (average * i + data) / (i + 1)
        if i == num_images - 1:
            consumer.send(average)
        i += 1


@coroutine
def sinograms(num_radiographs, consumer):
    """
    sinograms(num_radiographs, consumer)

    Convert *num_radiographs* into sinograms and send them to *consumer*.
    The sinograms are sent every time a new radiograph arrives. If there
    is more than *num_radiographs* radiographs, the sinograms are rewritten
    in a ring-buffer fashion.
    """
    i = 0
    sinograms = None

    def is_compatible(radio_shape, sinos_shape):
        return radio_shape[0] == sinos_shape[0] and \
            radio_shape[1] == sinos_shape[2]

    while True:
        radiograph = yield

        if sinograms is None or i % num_radiographs == 0:
            sinograms = np.zeros((radiograph.shape[0],
                                  num_radiographs,
                                  radiograph.shape[1]))
        if not is_compatible(radiograph.shape, sinograms.shape):
            raise ValueError("Incompatible radiograph shape")

        sinograms[:, i % num_radiographs, :] = radiograph
        consumer.send(sinograms)

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
def backproject(center, consumer, fast=True):
    """
    backproject(center, consumer, fast=True)

    Filtered backprojection filter. The input is (sinograms, stop), where
    *sinograms* is a 3D volume of sinograms and *stop* is the number of
    projections already present in sinograms. Reconstructed 3D slices
    are sent co *consumer*. If *fast* is True, try to use numpy for
    faster recontruction, however this approach demands much more memory.
    """
    slices = None
    batch = 100
    angle_step = None

    while True:
        sinograms = yield

        if slices is None:
            # Initialize reconstruction based on the sinograms shape
            width = sinograms.shape[2]
            num_slices = sinograms.shape[0]
            num_projections = sinograms.shape[1]
            y_points, x_points = np.mgrid[
                -center:width - center, -center:width - center]
            x_points = x_points.astype(np.float32)
            y_points = y_points.astype(np.float32)
            ramp_filter = get_ramp_filter(width)
            angle_step = np.pi / num_projections

        slices = np.zeros((num_slices, width, width), dtype=np.float32)

        # 1D sinogram filter
        sinograms_ft = np.fft.fft(sinograms)
        filtered = sinograms_ft * ramp_filter
        filtered = np.fft.ifft(filtered).real.astype(np.float32)

        for start in range(0, num_projections, batch):
            stop = start + batch
            if stop > num_projections:
                stop = num_projections
            backproject_algorithm(
                filtered, center, slices, angle_step=angle_step,
                start_projection=start, end_projection=stop, x_points=x_points,
                y_points=y_points, fast=fast)

        consumer.send(slices)


class PickSlice(object):

    """Pick a slice from a 3D volume."""

    def __init__(self, index):
        self.index = index

    @coroutine
    def __call__(self, consumer):
        """Pick a slice and send it to *consumer*."""
        while True:
            volume = yield
            consumer.send(volume[self.index:self.index+1])
