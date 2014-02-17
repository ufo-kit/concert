"""Imaging experiments usually conducted at synchrotrons."""
import os
import numpy as np
from concert.quantities import q
from concert.storage import write_tiff, create_directory
from concert.coroutines.sinks import write_images
from concert.experiments.base import Experiment as BaseExperiment


class Experiment(BaseExperiment):

    """
    Imaging experiment stores images acquired in acquisitions on disk
    automatically.
    """

    def __init__(self, acquisitions, directory_prefix, log=None, log_file_name="experiment.log",
                 writer=write_tiff):
        super(Experiment, self).__init__(acquisitions, directory_prefix, log=log,
                                         log_file_name=log_file_name)
        self.writer = writer
        self._writers = {}

    def acquire(self):
        """Run the experiment. Add writers to acquisitions dynamically."""
        try:
            for acq in self.acquisitions:
                self._attach_writer(acq)
                acq()
        finally:
            self._remove_writers()

    def _attach_writer(self, acquisition):
        """
        Attach a writer to an *acquisition* in order to store the images on
        disk automatically.
        """
        directory = os.path.join(self.directory, acquisition.name)
        create_directory(directory)
        prefix = os.path.join(directory, "frame_{:>05}")

        if acquisition not in self._writers:
            writer = write_images(writer=self.writer, prefix=prefix)
            # The launcher returns the original writer even if the acquisition
            # is invoked multiple times. In that case the images from every run
            # are appended to the original ones.
            launcher = lambda: writer
            acquisition.consumers.append(launcher)
            self._writers[acquisition] = launcher

    def _remove_writers(self):
        """
        Cleanup the image writing in order to be able to write images into
        the current directory during the next scan.
        """
        for acq, launcher in self._writers.items():
            acq.consumers.remove(launcher)
        self._writers = {}


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
