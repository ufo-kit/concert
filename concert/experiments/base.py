"""
An experiment can be run multiple times. The base :py:class:`.Experiment` takes
care of proper logging structure.
"""

import asyncio
import logging
import os
import time
from concert.coroutines.base import background, broadcast
from concert.coroutines.sinks import null
from concert.progressbar import wrap_iterable
from concert.base import check, Parameterizable, Parameter, Selection, State, StateError, transition


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

        a coroutine function which acquires the data, takes no arguments, can be None.

    """

    def __init__(self, name, producer, consumers=None, acquire=None):
        self.name = name
        self.producer = producer
        self.consumers = [] if consumers is None else consumers
        # Don't bother with checking this for None later
        if acquire and not asyncio.iscoroutinefunction(acquire):
            raise TypeError('acquire must be a coroutine function')
        self.acquire = acquire

    @background
    async def __call__(self):
        """Run the acquisition, i.e. acquire the data and connect the producer and consumers."""
        LOG.debug(f"Running acquisition '{self.name}'")
        consumers = self.consumers
        if not consumers:
            LOG.debug(f"`{self.name}' has no consumers, using null")
            consumers = [null]

        if self.acquire:
            await self.acquire()

        coros = broadcast(self.producer(), *consumers)
        await asyncio.gather(*coros, return_exceptions=False)

    def __repr__(self):
        return "Acquisition({})".format(self.name)


class Experiment(Parameterizable):

    """
    Experiment base class. An experiment can be run multiple times with the output data and log
    stored on disk. You can prepare every run by :meth:`.prepare` and finsh the run by
    :meth:`.finish`. These methods do nothing by default. They can be useful e.g. if you need to
    reinitialize some experiment parts or want to attach some logging output.

    .. py:attribute:: acquisitions
        :noindex:

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

        if separate_scans and walker:
            # The data is not supposed to be overwritten, so find an iteration which
            # hasn't been used yet
            while self.walker.exists(self._name_fmt.format(self._iteration)):
                self._iteration += 1

    async def _get_iteration(self):
        return self._iteration

    async def _set_iteration(self, iteration):
        self._iteration = iteration

    async def _get_separate_scans(self):
        return self._separate_scans

    async def _set_separate_scans(self, separate_scans):
        self._separate_scans = separate_scans

    async def _get_name_fmt(self):
        return self._name_fmt

    async def _set_name_fmt(self, fmt):
        self._name_fmt = fmt

    async def _get_log_level(self):
        return logging.getLevelName(self.log.getEffectiveLevel()).lower()

    async def _set_log_level(self, level):
        self.log.setLevel(level.upper())

    async def prepare(self):
        """Gets executed before every experiment run."""
        pass

    async def finish(self):
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

    async def acquire(self):
        """
        Acquire data by running the acquisitions. This is the method which implements
        the data acquisition and should be overriden if more functionality is required,
        unlike :meth:`~.Experiment.run`.
        """
        for acq in wrap_iterable(self._acquisitions):
            if await self.get_state() != 'running':
                break
            await acq()

    @background
    @check(source=['standby', 'error'], target='standby')
    @transition(immediate='running', target='standby')
    async def run(self):
        start_time = time.time()
        handler = None
        iteration = await self.get_iteration()
        separate_scans = await self.get_separate_scans()

        try:
            if self.walker:
                if separate_scans:
                    self.walker.descend((await self.get_name_fmt()).format(iteration))
                if os.path.exists(self.walker.current):
                    # We might have a dummy walker which doesn't create the directory
                    handler = logging.FileHandler(os.path.join(self.walker.current,
                                                               'experiment.log'))
                    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s '
                                                  '- %(message)s')
                    handler.setFormatter(formatter)
                    self.log.addHandler(handler)
            self.log.info(await self.info_table)
            LOG.debug('Experiment iteration %d start', iteration)
            await self.prepare()
            await self.acquire()
        except asyncio.CancelledError:
            # This is normal, no special state needed -> standby
            LOG.warn('Experiment cancelled')
        except Exception as e:
            # Something bad happened and we can't know what, so set the state to error
            LOG.warn(f"Error `{e}' while running experiment")
            raise StateError('error', msg=str(e))
        except KeyboardInterrupt:
            LOG.warn('Experiment cancelled by keyboard interrupt')
            self._state_value = 'standby'
            raise
        finally:
            try:
                await self.finish()
            except Exception as e:
                LOG.warn(f"Error `{e}' while finalizing experiment")
                raise StateError('error', msg=str(e))
            finally:
                if separate_scans and self.walker:
                    self.walker.ascend()
                LOG.debug('Experiment iteration %d duration: %.2f s',
                          iteration, time.time() - start_time)
                if handler:
                    handler.close()
                    self.log.removeHandler(handler)
                await self.set_iteration(iteration + 1)


class AcquisitionError(Exception):
    """Acquisition-related exceptions."""
    pass


class ExperimentError(Exception):
    """Experiment-related exceptions."""
    pass
