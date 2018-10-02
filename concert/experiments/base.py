"""
An experiment can be run multiple times. The base :py:class:`.Experiment` takes
care of proper logging structure.
"""

import logging
import time
from concert.async import async


LOG = logging.getLogger(__name__)


class Acquisition(object):

    """
    An acquisition acquires data, gets it and sends it to consumers.

    .. py:attribute:: producer

        a callable with no arguments which returns a generator yielding data items once called.

    .. py:attribute:: consumers

        a list of callables with no arguments which return a coroutine consuming the data once
        started, can be empty.

    .. py:attribute:: acquire

        a callable which acquires the data, takes no arguments, can be None.

    """

    def __init__(self, name, producer, consumers=None, acquire=None):
        self.name = name
        self.producer = producer
        self.consumers = [] if consumers is None else consumers
        # Don't bother with checking this for None later
        self.acquire = acquire if acquire else lambda: None
        self._aborted = False

    def connect(self):
        """Connect producer with consumers."""
        self._aborted = False
        started = []
        for not_started in self.consumers:
            started.append(not_started())

        for item in self.producer():
            if self._aborted:
                LOG.info("Acquisition '%s' aborted", self.name)
                break
            for consumer in started:
                consumer.send(item)

    def abort(self):
        self._aborted = True

    def __call__(self):
        """Run the acquisition, i.e. acquire the data and connect the producer and consumers."""
        LOG.debug("Running acquisition '{}'".format(self))

        if self.acquire:
            self.acquire()

        self.connect()

    def __repr__(self):
        return "Acquisition({})".format(self.name)


class Experiment(object):

    """
    Experiment base class. An experiment can be run multiple times with the output data and log
    stored on disk. You can prepare every run by :meth:`.prepare` and finsh the run by
    :meth:`.finish`. These methods do nothing by default. They can be useful e.g. if you need to
    reinitialize some experiment parts or want to attach some logging output.

    .. py:attribute:: acquisitions

        A list of acquisitions this experiment is composed of

    .. py:attribute:: walker

       A :class:`concert.storage.Walker` descends to a data set specific for every run if given

    .. py:attribute:: separate_scans

        If True, *walker* does not descend to data sets based on specific runs

    .. py:attribute:: name_fmt

        Since experiment can be run multiple times each iteration will have a separate entry
        on the disk. The entry consists of a name and a number of the current iteration, so the
        parameter is a formattable string.

    """

    def __init__(self, acquisitions, walker=None, separate_scans=True, name_fmt='scan_{:>04}'):
        self._acquisitions = []
        for acquisition in acquisitions:
            self.add(acquisition)
        self.walker = walker
        self.separate_scans = separate_scans
        self.name_fmt = name_fmt
        self.iteration = 1
        self._aborted = False

        if self.separate_scans and self.walker:
            # The data is not supposed to be overwritten, so find an iteration which
            # hasn't been used yet
            while self.walker.exists(self.name_fmt.format(self.iteration)):
                self.iteration += 1

    def prepare(self):
        """Gets executed before every experiment run."""
        pass

    def finish(self):
        """Gets executed after every experiment run."""
        pass

    @property
    def acquisitions(self):
        """Acquisitions is a read-only attribute which has to be manipulated by explicit methods
        provided by this class.
        """
        return tuple(self._acquisitions)

    def add(self, acquisition):
        """
        Add *acquisition* to the acquisition list and make it accessible as
        an attribute::

            frames = Acquisition(...)
            experiment.add(frames)
            # This is possible
            experiment.frames
        """
        self._acquisitions.append(acquisition)
        setattr(self, acquisition.name, acquisition)

    def remove(self, acquisition):
        """Remove *acquisition* from experiment."""
        self._acquisitions.remove(acquisition)
        delattr(self, acquisition.name)

    def swap(self, first, second):
        """
        Swap acquisition *first* with *second*. If there are more occurences
        of either of them then the ones which are found first in the acquisitions
        list are swapped.
        """
        if first not in self._acquisitions or second not in self._acquisitions:
            raise ValueError("Both acquisitions must be part of the experiment")

        first_index = self._acquisitions.index(first)
        second_index = self._acquisitions.index(second)
        self._acquisitions[first_index] = second
        self._acquisitions[second_index] = first

    def get_acquisition(self, name):
        """
        Get acquisition by its *name*. In case there are more like it, the first
        one is returned.
        """
        for acq in self._acquisitions:
            if acq.name == name:
                return acq
        raise ExperimentError("Acquisition with name `{}' not found".format(name))

    def abort(self):
        LOG.info('Experiment aborted')
        self._aborted = True
        for acq in self.acquisitions:
            acq.abort()

    def acquire(self):
        """
        Acquire data by running the acquisitions. This is the method which implements
        the data acquisition and should be overriden if more functionality is required,
        unlike :meth:`~.Experiment.run`.
        """
        for acq in self._acquisitions:
            if self._aborted:
                break
            acq()

    @async
    def run(self):
        """
        run()

        Compute the next iteration and run the :meth:`~.base.Experiment.acquire`.
        """
        start_time = time.time()
        self._aborted = False
        LOG.debug('Experiment iteration %d start', self.iteration)
        if self.separate_scans and self.walker:
            self.walker.descend(self.name_fmt.format(self.iteration))

        try:
            self.prepare()
            self.acquire()
            self.finish()
        except:
            LOG.exception('Error while running experiment')
            raise
        finally:
            if self.separate_scans and self.walker:
                self.walker.ascend()
            LOG.debug('Experiment iteration %d duration: %.2f s',
                      self.iteration, time.time() - start_time)
            self.iteration += 1


class ExperimentError(Exception):
    """Experiment-related exceptions."""
    pass
