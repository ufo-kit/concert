try:
    import Queue as queue
except ImportError:
    import queue
import logging
import numpy as np
from concert.imageprocessing import (backproject,
                                     get_backprojection_norm,
                                     get_ramp_filter)
from concert.async import threaded
from concert.coroutines import coroutine
from concert.imageprocessing import flat_correct as make_flat_correct


LOG = logging.getLogger(__name__)


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
def make_sinograms(num_radiographs, consumer):
    """
    make_sinograms(num_radiographs, consumer)

    Convert *num_radiographs* into sinograms and send them to *consumer*.
    """
    i = 0
    sinograms = None

    def is_compatible(radio_shape, sinos_shape):
        return radio_shape[0] == sinos_shape[0] and \
            radio_shape[1] == sinos_shape[2]

    while True:
        radiograph = yield

        if sinograms is None:
            sinograms = np.empty((radiograph.shape[0],
                                  num_radiographs,
                                  radiograph.shape[1]))
        if i < num_radiographs:
            if not is_compatible(radiograph.shape, sinograms.shape):
                raise ValueError("Incompatible radiograph shape")

            sinograms[:, i, :] = np.copy(radiograph)
            if i == num_radiographs - 1:
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
def backprojector(row_number, center, consumer, num_projs=None,
                  angle_step=None, nth_column=1, nth_projection=1,
                  callback=None, fast=True):
    """
    backprojector(row_number, center, consumer, num_projs=None,\
                      angle_step=None, nth_column=1, nth_projection=1,\
                                        callback=None, fast=True)

    Online filtered backprojection. Get a radiograph, extract row
    *row_number*, backproject it and add to the so far computed slice.
    *center* is the center of rotation, *num_projs* determine how many
    projections to expect, *angle_step* is the angular rotation step
    between two projections. If it is None, it is calculated
    automatically from *num_projs*. *nth_column* determines the downsampling
    of incoming data, every n-th column of the sinogram row will be
    taken into account, similar for *nth_projection*, just says which
    angles are skipped, *consumer* is a generator to which the each
    freshly computed slice is sent and *callback* is a function which
    is called when the *num_projs* rows are added to the slice. The
    slice is sent to the callback function as its only arguemnt. If
    *fast* is True, try to use a fast backprojection algorithm, in case it
    cannot be employed fall back to the slow one automatically.

    The backprojection routine runs in a separate thread in order not
    to stall possible frame grabbing.
    """
    i = 0
    center /= float(nth_column)
    if angle_step is None and num_projs is None:
        raise ValueError("One of angle_step or num_projs must be specified")
    angle_step = np.pi / num_projs if angle_step is None else angle_step
    sino_queue = queue.Queue()

    @threaded
    def backprojection_dispatcher(fast):
        """
        Dispatch update of the slice. Accept sinogram row and projection
        angle from a queue and add the sinogram row to the resulting
        slice.

        The backprojection operates on 32-bit float numbers and uses
        nearest-neighbor interpolation.
        """
        result = None
        j = 0

        while True:
            data = sino_queue.get()
            if data is None:
                # Backprojection finished
                if callback is not None:
                    callback(result)
                break
            else:
                # Items put in the queue have the form [sino_row, theta]
                row, angle_index = data

            if result is None:
                # Initialize reconstruction based on the sinogram row shape
                width = row.shape[0]
                sinogram = np.empty((1, width), dtype=np.float32)

                result = np.zeros((width, width), dtype=np.float32)
                y_points, x_points = np.mgrid[-center:width - center,
                                              -center:width - center]
                x_points = x_points.astype(np.float32)
                y_points = y_points.astype(np.float32)
                ramp_filter = get_ramp_filter(width)

            # 1D sinogram filter
            row = np.fft.ifft(np.fft.fft(row) * ramp_filter).\
                real.astype(np.float32)

            # Make the sinogram row 2D to comply with reconstruction
            # algorithms
            sinogram[0] = row
            if fast:
                try:
                    result += backproject(sinogram, center,
                                          angle_step=angle_step,
                                          start_projection=angle_index,
                                          x_points=x_points, y_points=y_points,
                                          fast=fast)
                except MemoryError:
                    LOG.debug("Not enough memory, falling back to slow " +
                              "backprojection algorithm")
                    fast = False
            if not fast:
                result += backproject(sinogram, center,
                                      angle_step=angle_step,
                                      start_projection=angle_index,
                                      x_points=x_points, y_points=y_points,
                                      fast=fast)

            consumer.send(result * get_backprojection_norm(j + 1))

            j += 1

    backprojection_dispatcher(fast)

    while True:
        image = yield
        if num_projs is None or i < num_projs:
            if i % nth_projection == 0:
                # Calculate in separate thread in order not to stall
                # frame grabbing
                # Extract row which we are interested in
                row = image[row_number, :]
                sino_queue.put((row[::nth_column], i))
            if num_projs is not None and i == num_projs - 1:
                # Maximum projections reached, stop reconstruction
                sino_queue.put(None)
        i += 1
