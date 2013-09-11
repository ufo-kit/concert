"""Measures which work on the 2D image plane."""
import numpy as np


class Area(object):

    r"""
    Area measure how much of the whole image is covered by an object.
    The return value is the fraction of object pixels and the whole
    area, :math:`n = object_pixels / (width \cdot height)`.

    .. py:attribute:: func

        a callable *func(radio, \*args, \*\*kwargs)*, where
        *radio* is a radiograph to be analyzed

    .. py:attribute:: args

        *func* arguments tuple

    .. py:attribute:: kwargs

        *func* keyword arguments dictionary

    """

    def __init__(self, func, args=None, kwargs=None):
        self.func = func
        self.args = () if args is None else args
        self.kwargs = {} if kwargs is None else kwargs

    def __call__(self, radio):
        object_pixels = self.func(radio, *self.args, **self.kwargs)

        return float(object_pixels) / (radio.shape[0] * radio.shape[1])


class SimpleArea(Area):

    """
    An area measure with object segmentation by a threshold. If two
    flat fields *flat_1* and *flat_2* are given, then the minimum
    of their subtraction :math:`min(flat_1, flat_2)` approximates
    the maximum grey value considered a background. The values
    from the radiograph and flat field subtraction below the
    minimum are considered an object.
    """

    def __init__(self, flat_1, flat_2):
        super(SimpleArea, self).__init__(self.get_object_pixels)

        # Threshold
        self.thr = np.min(flat_1 - flat_2)
        # Average the two flats to suppress the noise a little
        self.flat = (flat_1 + flat_2) / 2.0

    def get_object_pixels(self, radio):
        """Get object pixels from a radiograph *radio*."""
        radio = radio - self.flat

        return len(np.where(radio < self.thr)[0])
