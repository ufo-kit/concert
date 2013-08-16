"""
The :mod:`.alignment` module provides pre-defined functions built from the
optimization classes to execute common alignment procedures.
"""
import numpy as np
from concert.quantities import q
from concert.optimization.algorithms import halver
from concert.optimization.optimizers import Maximizer


def focus(camera, motor, measure=np.std):
    """
    Focus *camera* by moving *motor*. *measure* is a callable that computes a
    scalar that has to be maximized from an image taken with *camera*.

    This function is returning a future encapsulating the focusing event. Note,
    that the camera is stopped from recording as soon as the optimal position
    is found.
    """
    opts = {'initial_step': 10 * q.mm, # we should guess this from motor limits
            'epsilon': 5e-3 * q.mm}

    def get_measure():
        frame = camera.grab()
        return measure(frame)

    maximizer = Maximizer(motor['position'],
                          get_measure,
                          halver, alg_kwargs=opts)

    camera.start_recording()
    f = maximizer.run()
    f.add_done_callback(lambda: camera.stop_recording())
    return f
