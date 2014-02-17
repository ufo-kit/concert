"""
An experiment can be run multiple times. The base :py:class:`.Experiment`
takes care of proper logging structure.
"""

import os
import re
import logging
from logging import FileHandler, Formatter
from concert.async import async
from concert.storage import create_directory
from concert.coroutines import broadcast, inject


LOG = logging.getLogger(__name__)


class Acquisition(object):

    """
    An acquisition object connects data generator to consumers.

    .. py:attribute:: generator_caller

        a callable which returns a generator once called

    .. py:attribute:: consumer_callers

        a list of callables which return a coroutine once started

    """

    def __init__(self, name, generator_caller, consumer_callers=None):
        self.name = name
        self.generator = generator_caller
        self.consumers = [] if consumer_callers is None else consumer_callers

    def __call__(self):
        """Run the acquisition, i.e. connect the producer and consumer."""
        LOG.debug("Running acquisition '{}'".format(self))

        started = []
        for not_started in self.consumers:
            started.append(not_started())

        inject(self.generator(), broadcast(*started))

    def __repr__(self):
        return "Acquisition({})".format(self.name)


class Experiment(object):

    r"""
    Experiment base class. An experiment can be run multiple times
    with logging output saved on disk. The log from every
    :meth:`~.Experiment.run` is saved in the current experiment
    directory given by *directory_prefix*.

    .. py:attribute:: acquisitions

        A list of acquisitions this experiment is composed of

    .. py:attribute:: directory_prefix

       Directory prefix is either a formattable string in which case the
       at each experiment run a new directory given by the prefix and
       the current iteration is created. If the *directory_prefix* is a
       simple string then the individual experiment runs are stored in
       its subdirectories starting with scan\_ and suffixed by the run
       iteration.

    .. py:attribute:: log

        A logger to which a file handler will be attached in order to
        store the log output in the current directory

    .. py:attribute:: log_file_name

        Log file name used for storing logging information.

    """

    def __init__(self, acquisitions, directory_prefix, log=None,
                 log_file_name="experiment.log"):
        self.acquisitions = acquisitions
        self.directory_prefix = directory_prefix
        self.log = log
        self.log_file_name = log_file_name
        pattern = re.compile(".*\{.*\}.*")
        if pattern.match(self.directory_prefix) is None:
            self.directory_prefix = os.path.join(self.directory_prefix, "scan_{:>03}")
        self._file_stream = None
        self.iteration = 1

    def swap(self, first, second):
        """
        Swap acquisition *first* with *second*. If there are more occurences
        of either of them then the ones which are found first in the acquisitions
        list are swapped.
        """
        if first not in self.acquisitions or second not in self.acquisitions:
            raise ValueError("Both acquisitions must be part of the experiment")

        first_index = self.acquisitions.index(first)
        second_index = self.acquisitions.index(second)
        tmp = first
        self.acquisitions[first_index] = second
        self.acquisitions[second_index] = tmp

    @property
    def directory(self):
        """Current directory for running the experiment."""
        return self.directory_prefix.format(self.iteration)

    def _create_stream_handler(self):
        """
        Create file stream handler for given scan to log information
        to the current directory.
        """
        path = os.path.join(self.directory, self.log_file_name)
        self._file_stream = FileHandler(path)
        self._file_stream.setLevel(logging.INFO)
        formatter = Formatter("[%(asctime)s] %(levelname)s: %(name)s: %(message)s")
        self._file_stream.setFormatter(formatter)
        self.log.addHandler(self._file_stream)

    def acquire(self):
        """
        Acquire data by running the acquisitions. This is the method which implements
        the data acquisition and should be overriden if more functionality is required,
        unlike :meth:`~.Experiment.run`.
        """
        for acq in self.acquisitions:
            acq()

    @async
    def run(self):
        """
        run()

        Create current directory, attach logging output to file and run the
        :meth:`~.base.Experiment.acquire`. After the run is complete the logging
        is cleaned up automatically. This method should *not* be overriden.
        """
        # Create directory for next scan
        while os.path.exists(self.directory):
            # Iterate until an unused directory has been found
            self.iteration += 1
        create_directory(self.directory)

        # Initiate new logger for this scan
        if self.log:
            self._create_stream_handler()

        try:
            self.acquire()
        finally:
            if self.log:
                self._file_stream.close()
                self.log.removeHandler(self._file_stream)


class ExperimentError(Exception):
    pass
