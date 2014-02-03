"""
Imaging positioner which enables us to move around by specyfying distances
in the number of pixels.
"""
import numpy as np
from concert.quantities import q
from concert.devices.positioners import base


class Positioner(base.Positioner):

    """
    A positioner which takes into account a detector with some pixel size.
    This way the user can specify the movement in pixels.
    """

    def __init__(self, axes, detector, position=None):
        super(Positioner, self).__init__(axes, position=position)
        self.detector = detector

    def move(self, position):
        """Move by specified *position* which can be given in meters or pixels."""
        # Is there a better way to check units?
        if str(position.units) == 'pixel':
            mag = position.magnitude
            if mag[-1] != 0 and not np.isnan(mag[-1]):
                raise base.PositionerError('Cannot set \'z\' coordinate position in pixels.')

            physical_position = [0, 0, 0]
            width = self.detector.pixel_width.to(q.m).magnitude
            height = self.detector.pixel_height.to(q.m).magnitude
            physical_position[0] = position[0].magnitude * width
            physical_position[1] = position[1].magnitude * height
            physical_position *= q.m
        else:
            physical_position = position

        return super(Positioner, self).move(physical_position)
