import numpy as np
from concert.processes.base import Feedback, Scanner


class FrameMean(Feedback):
    """Calculate the noise level depending on a variable parameter.

    *camera* is a camera object that supports :meth:`.grab`, *param_name*
    is a parameter name of *camera* that should be varied according.
    """
    def __init__(self, camera):
        self.camera = camera

    def __call__(self):
        frame = self.camera.grab()

        # Should this be over whole frame or just those pixels above 0?
        return np.mean(frame)


class PhotonTransfer(Feedback):
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
