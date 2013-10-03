try:
    import Queue as queue
except ImportError:
    import queue

import logbook
import numpy as np
from concert.helpers import dispatcher, coroutine, threaded
from concert.storage import write_tiff
from concert.imageprocessing import backproject, get_backprojection_norm,\
    get_ramp_filter, flat_correct as make_flat_correct


LOG = logbook.Logger(__name__)


SINOGRAMS_FULL = "sinos-full"


@coroutine
def write_images(writer=write_tiff, prefix="image_{:>05}"):
    """
    write_images(writer, prefix="image_{:>05}")

    Write images on disk with specified *writer* and file name *prefix*.
    *writer* is a callable with the following nomenclature::

        writer(file_name_prefix, data)

    The file extension needs to be applied by a particular writer.
    """
    i = 0

    while True:
        data = yield
        writer(prefix.format(i), data)
        i += 1


@coroutine
def generate_sinograms(sinograms):
    """
    generate_sinograms(sinograms)

    Generate *sinograms* from radiographs. *sinograms* is a 3D numpy
    array to which a radiograph will be inserted. The shape of
    the array is (num_sinograms, slice_height, slice_width).
    The number of sinograms must be a divisor of the radiograph height.
    If the number of sinograms is lower, then every :math:`i`-th row of
    a radiograph is taken into account, i.e.
    :math:`i \cdot num\_sinograms = radio\_height`.
    """
    i = 0
    ith = None

    while True:
        radiograph = yield
        if i < sinograms.shape[0]:
            if radiograph.shape[0] % sinograms.shape[0] != 0 or \
                radiograph.shape[0] < sinograms.shape[0] or \
                    radiograph.shape[1] != sinograms.shape[1]:
                raise ValueError("Incompatible radiograph shape")
            if ith is None:
                ith = radiograph.shape[0] / sinograms.shape[0]
            sinograms[:, i, :] = radiograph[::ith, :]
        else:
            dispatcher.send(sinograms, SINOGRAMS_FULL)
        i += 1


class ImageAverager(object):

    """Average images in a coroutine without knowing how many will come."""

    def __init__(self):
        self.average = None

    @coroutine
    def average_images(self):
        """Average images as they come."""
        i = 0
        while True:
            data = yield
            if self.average is None:
                self.average = np.zeros_like(data, dtype=np.float32)
            self.average = (self.average * i + data) / (i + 1)
            i += 1


@coroutine
def flat_correct(consumer, dark, flat):
    """
    Flat correction intermediate coroutine. It takes a *dark*, a *flat*,
    gets a radiograph from a generator, calculates flat corrected image
    and sends it forward to *consumer*.
    """
    while True:
        radio = yield
        consumer.send(make_flat_correct(radio, flat, dark))


@coroutine
def backprojector(row_number, center, num_projs=None, angle_step=None,
                  nth_column=1, nth_projection=1, consumer=None,
                  callback=None, fast=True):
    """
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

            if consumer is not None:
                # If there is a consumer, update it with the newest result
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
