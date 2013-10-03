from concert.helpers import dispatcher, coroutine
from concert.storage import write_tiff


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
        import numpy as np

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
        consumer.send((radio - dark) / (flat - dark))
