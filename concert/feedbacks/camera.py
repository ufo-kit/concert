"""Camera-related feedbacks to be used with a :class:`.Scanner` process.

To evaluate the mean according to different exposure times you can setup a
process like this ::

    camera = Camera()
    feedback = FrameMean(camera)
    scanner = Scanner(camera['exposure-time'], feedback)
    scanner.minimum = 1 * q.ms
    scanner.maximum = 1 * q.s

    x, y = scanner.run().result()
"""
import numpy as np
from concert.quantities import q
from concert.base import Parameter


class FrameMean(Parameter):

    """Grab a frame and calculate the mean.

    *camera* is a camera object that supports :meth:`.grab`, *param_name*
    is a parameter name of *camera* that should be varied according.
    """

    def __init__(self, camera):
        self.camera = camera
        super(FrameMean, self).__init__('frame-mean', unit=q.counts)

    def __call__(self):
        frame = self.camera.grab()

        # Should this be over whole frame or just those pixels above 0?
        return np.mean(frame)


class PhotonTransfer(object):

    """Calculate the photon transfer according to procedure described by M.
    Caselle and F. Beckmann.

    *camera* is a camera object that supports the :meth:`.grab`, *dark_frame*
    is an array-like with the same dimensions as the camera frames.
    """

    def __init__(self, camera, dark_frame):
        self.camera = camera
        self.dark_frame = dark_frame

    def __call__(self):
        frame = self.camera.grab()
        return np.log(np.sum(np.abs(frame - self.dark_frame)))
