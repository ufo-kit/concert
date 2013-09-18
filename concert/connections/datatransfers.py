"""One-to-one and one-to-many data transfers."""
from concert.base import coroutine
from concert.asynchronous import dispatcher


SINOGRAMS_FULL = "sinos-full"


@coroutine
def multicast(*destinations):
    """
    multicast(*destinations)

    Provide data to all *destinations*.
    """
    while True:
        item = yield
        for destination in destinations:
            destination.send(item)


@coroutine
def write_images(writer, prefix="radio_"):
    """
    write_images(writer, prefix="radio_")

    Write images on disk with specified *writer* and file name *prefix*.
    *writer* is a callable with the following nomenclature::

        writer(file_name_prefix, data)

    The file extension needs to be applied by a particular writer.
    """
    i = 0

    while True:
        data = yield
        file_name_prefix = prefix.join("{:>06}".format(i))
        writer(file_name_prefix, data)
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
