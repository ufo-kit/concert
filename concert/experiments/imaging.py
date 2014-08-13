"""Imaging experiments usually conducted at synchrotrons."""
import numpy as np
from concert.quantities import q
from concert.experiments.base import Experiment as BaseExperiment


class Experiment(BaseExperiment):

    """
    Imaging experiment stores images acquired in acquisitions on disk
    automatically.
    """

    def acquire(self):
        """Run the experiment. Add writers to acquisitions dynamically."""
        for acq in self.acquisitions:
            writer = lambda: self.walker.write()
            self.walker.descend(acq.name)
            acq.consumers.append(writer)
            try:
                acq()
            finally:
                self.walker.ascend()
                acq.consumers.remove(writer)


def frames(num_frames, camera, callback=None):
    """
    A generator which takes *num_frames* using *camera*. *callback* is called
    after every taken frame.
    """
    if camera.state == 'recording':
        camera.stop_recording()

    camera['trigger_mode'].stash().join()
    camera.trigger_mode = camera.trigger_modes.SOFTWARE

    try:
        with camera.recording():
            for i in range(num_frames):
                camera.trigger()
                yield camera.grab()
                if callback:
                    callback()
    finally:
        camera['trigger_mode'].restore().join()


def tomo_angular_step(frame_width):
    """
    Get the angular step required for tomography so that every pixel of the frame
    rotates no more than one pixel per rotation step. *frame_width* is frame size in
    the direction perpendicular to the axis of rotation.
    """
    return np.arctan(2.0 / frame_width.magnitude) * q.rad


def tomo_projections_number(frame_width):
    """
    Get the minimum number of projections required by a tomographic scan in
    order to provide enough data points for every distance from the axis of
    rotation. The minimum angular step is
    considered to be needed smaller than one pixel in the direction
    perpendicular to the axis of rotation. The number of pixels in this
    direction is given by *frame_width*.
    """
    return int(np.ceil(np.pi / tomo_angular_step(frame_width)))


def tomo_max_speed(frame_width, frame_rate):
    """
    Get the maximum rotation speed which introduces motion blur less than one
    pixel. *frame_width* is the width of the frame in the direction
    perpendicular to the rotation and *frame_rate* defines the time required
    for recording one frame.

    _Note:_ frame rate is required instead of exposure time because the
    exposure time is usually shorter due to the camera chip readout time.
    We need to make sure that by the next exposure the sample hasn't moved
    more than one pixel from the previous frame, thus we need to take into
    account the whole frame taking procedure (exposure + readout).
    """
    return tomo_angular_step(frame_width) * frame_rate
