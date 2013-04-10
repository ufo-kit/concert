import quantities as q
import numpy as np
from concert.processes.base import Scanner


class NoiseScan(Scanner):
    """Calculate the noise level depending on a variable parameter.

    *camera* is a camera object that supports :meth:`.grab`, *param_name*
    is a parameter name of *camera* that should be varied according.
    """
    def __init__(self, camera, param_name):
        super(NoiseScan, self).__init__(camera[param_name])

        self._camera = camera
        self.minimum = 0.01 * q.s
        self.maximum = 1.0 * q.s
        self.intervals = 50

    def evaluate(self):
        frame = self._camera.grab()

        # Should this be over whole frame or just those pixels above 0?
        return np.mean(frame)
