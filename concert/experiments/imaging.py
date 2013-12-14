"""Imaging experiments usually conducted at synchrotrons."""
"""Imaging experiments usually conducted at synchrotrons."""
import os
import logging
from concert.storage import write_tiff, create_directory
from concert.coroutines.sinks import write_images
from concert.experiments.base import Experiment as BaseExperiment


LOG = logging.getLogger(__name__)


DARKS = "darks"
FLATS = "flats"
RADIOS = "radios"


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
