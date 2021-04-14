"""
An experiment can be run multiple times. The base :py:class:`.Experiment` takes
care of proper logging structure.
"""

import logging
import os
import time
from concert.casync import casync
from concert.progressbar import wrap_iterable
from concert.base import Parameterizable, Parameter, Selection, State, check


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

    @casync
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


class Experiment(Parameterizable):

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

    iteration = Parameter()
    separate_scans = Parameter()
    name_fmt = Parameter()
    state = State(default='standby')
    log_level = Selection(['critical', 'error', 'warning', 'info', 'debug'])

    def __init__(self, acquisitions, walker=None, separate_scans=True, name_fmt='scan_{:>04}'):
        self._acquisitions = []
        for acquisition in acquisitions:
            self.add(acquisition)
        self.walker = walker
        self._separate_scans = separate_scans
        self._name_fmt = name_fmt
        self._iteration = 1
        self.log = LOG
        super(Experiment, self).__init__()

        if self.separate_scans and self.walker:
            # The data is not supposed to be overwritten, so find an iteration which
            # hasn't been used yet
            while self.walker.exists(self.name_fmt.format(self.iteration)):
                self.iteration += 1

    def _get_iteration(self):
        return self._iteration

    def _set_iteration(self, iteration):
        self._iteration = iteration

    def _get_separate_scans(self):
        return self._separate_scans

    def _set_separate_scans(self, separate_scans):
        self._separate_scans = separate_scans

    def _get_name_fmt(self):
        return self._name_fmt

    def _set_name_fmt(self, fmt):
        self._name_fmt = fmt

    def _get_log_level(self):
        return logging.getLevelName(self.log.getEffectiveLevel()).lower()

    def _set_log_level(self, level):
        self.log.setLevel(level.upper())

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

    @casync
    @check(source=['running'], target=['standby'])
    def abort(self):
        self._state_value = 'aborting'
        LOG.info('Experiment aborted')
        for acq in self.acquisitions:
            acq.abort().join()
        self._state_value = 'standby'

    def acquire(self):
        """
        Acquire data by running the acquisitions. This is the method which implements
        the data acquisition and should be overriden if more functionality is required,
        unlike :meth:`~.Experiment.run`.
        """
        for acq in wrap_iterable(self._acquisitions):
            if self.state != 'running':
                break
            acq()

    @casync
    @check(source=['standby', 'error'], target='standby')
    def run(self):
        """
        run()

        Compute the next iteration and run the :meth:`~.base.Experiment.acquire`.
        """
        self._state_value = 'running'
        start_time = time.time()
        handler = None

        try:
            if self.walker:
                if self.separate_scans:
                    self.walker.descend(self.name_fmt.format(self.iteration))
                if os.path.exists(self.walker.current):
                    # We might have a dummy walker which doesn't create the directory
                    handler = logging.FileHandler(os.path.join(self.walker.current,
                                                               'experiment.log'))
                    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s '
                                                  '- %(message)s')
                    handler.setFormatter(formatter)
                    self.log.addHandler(handler)
            self.log.info(self)
            LOG.debug('Experiment iteration %d start', self.iteration)
            self.prepare()
            self.acquire()
        except:
            self._state_value = 'error'
            LOG.exception('Error while running experiment')
            raise
        finally:
            try:
                self.finish()
            except:
                self._state_value = 'error'
                LOG.exception('Error while running experiment')
                raise
            finally:
                if self.separate_scans and self.walker:
                    self.walker.ascend()
                LOG.debug('Experiment iteration %d duration: %.2f s',
                          self.iteration, time.time() - start_time)
                if handler:
                    handler.close()
                    self.log.removeHandler(handler)
                self.iteration += 1

        self._state_value = 'standby'


class ExperimentError(Exception):
    """Experiment-related exceptions."""
    pass
