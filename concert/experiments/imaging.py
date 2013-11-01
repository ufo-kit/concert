"""Imaging experiments usually conducted at synchrotrons."""
import os
import logging
from concert.storage import write_tiff, create_directory
from concert.coroutines import write_images, ImageAverager, flat_correct as \
    do_flat_correct
from concert.helpers import inject, broadcast
from concert.coroutines import null
from concert.experiments.base import Experiment


LOG = logging.getLogger(__name__)


DARKS = "darks"
FLATS = "flats"
RADIOS = "radios"


class Radiography(Experiment):

    """
    An abstract radiography experiment. The user needs to provide the
    functionality by providing the acquisition methods::

        acquire_darks()
        acquire_flats()
        acquire_radios()

    and overriding the respective process methods. If *flat_correct* is
    True, the radiographs are first flat corrected
    and then send to *process_radios*.  *writer* specifies image writer
    which will handle the image storage.
    Every image is stored in a subdirectory based on its type, moreover it is
    stored in a particular scan number subdirectory if the experiment should
    store scans in separate subdirectories, i.e.::

        root_directory/[scan_directory/]darks
        root_directory/[scan_directory/]flats
        root_directory/[scan_directory/]radios

    directories exist after the run and are filled with images of a particular
    type. Note that radiographs are stored as they are taken without
    flat correction even if *flat_correct* is True.
    """

    def __init__(self, root_directory, iteration=1,
                 log_file_name="experiment.log", flat_correct=False,
                 writer=write_tiff):
        super(Radiography, self).__init__(self.execute, root_directory,
                                          iteration=iteration,
                                          log_file_name=log_file_name)
        self.flat_correct = flat_correct
        self.writer = writer
        if self.flat_correct and self.__class__.process_radios == \
                Radiography.process_radios:
            LOG.warn("Flat correction requested but radiographs" +
                     " processing functionality not provided, hence" +
                     " this setting has no effect.")

    def process_darks(self):
        """
        Process dark fields. By default this method does nothing. It has to
        be a coroutine.
        """
        return null()

    def process_flats(self):
        """
        Process flat fields. By default this method does nothing. It has to
        be a coroutine.
        """
        return null()

    def process_radios(self):
        """
        Process radiographs. By default this method does nothing. It has to
        be a coroutine.
        """
        return null()

    def execute(self):
        """
        Execute the experiment by acquiring and processing the respective image
        types. Every image type is processed in such a way, that the
        acquisition generator is connected to a broadcast which consists of the
        process coroutine and an additional image writer which will write the
        images to the disk.
        """
        # We do flat correction only if it is requested and radiographs
        # processing was specified.
        flat_correct = self.flat_correct and self.__class__.\
            process_radios != Radiography.process_radios

        if hasattr(self, "acquire_darks"):
            if flat_correct:
                dark_averager, _process_darks = \
                    self.add_averager(self.process_darks())
            else:
                _process_darks = self.process_darks()
            consumers = self.add_writer(_process_darks, DARKS)
            inject(self.acquire_darks(), consumers)

        if hasattr(self, "acquire_flats"):
            if flat_correct:
                flat_averager, _process_flats = \
                    self.add_averager(self.process_flats())
            else:
                _process_flats = self.process_flats()
            consumers = self.add_writer(_process_flats, FLATS)
            inject(self.acquire_flats(), consumers)

        if hasattr(self, "acquire_radios"):
            if flat_correct:
                _process_radios = do_flat_correct(self.process_radios(),
                                                  dark_averager.average,
                                                  flat_averager.average)
            else:
                _process_radios = self.process_radios()
            consumers = self.add_writer(_process_radios, RADIOS)
            inject(self.acquire_radios(), consumers)

    def add_averager(self, consumer):
        """Add an averager to the original *consumer*."""
        averager = None
        coroutines = []

        if self.flat_correct:
            averager = ImageAverager()
            coroutines.append(averager.average_images())
        if consumer is not None:
            coroutines.append(consumer)

        return averager, broadcast(*coroutines)

    def add_writer(self, process_images, image_type):
        """
        Add image writer to consumers. The resulting directory of the data
        is obtained by joining the *directory* and a subdirectory determined
        by *image_type*. *process_images* is an originally assigned
        image processing coroutine. *writer* specifies which image
        writer will be used, and thus also the file type.
        """
        directory = os.path.join(self.directory, image_type)
        create_directory(directory)
        prefix = os.path.join(directory, image_type[:-1] + "_{:>05}")
        coroutines = [write_images(writer=self.writer, prefix=prefix)]
        if process_images is not None:
            coroutines.append(process_images)

        return broadcast(*coroutines)
