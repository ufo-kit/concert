"""
An experiment can be run multiple times. The base :py:class:`.Experiment`
takes care of proper logging structure.
"""

import logging
from concert.async import async
from concert.coroutines.base import broadcast, inject


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
    Experiment base class. An experiment can be run multiple times with the output data and log
    stored on disk. You can prepare every run by :meth:`.prepare` and finsh the run by
    :meth:`.finish`. These methods do nothing by default. They can be useful e.g. if you need to
    reinitialize some experiment parts or want to attach some logging output.

    .. py:attribute:: acquisitions

        A list of acquisitions this experiment is composed of

    .. py:attribute:: walker

       A :class:`concert.storage.Walker` stores experimental data and
       logging output

    .. py:attribute:: name_fmt

        Since experiment can be run multiple times each iteration will have a separate entry
        on the disk. The entry consists of a name and a number of the current iteration, so the
        parameter is a formattable string.

    """

    def __init__(self, acquisitions, walker, name_fmt='scan_{:>04}'):
        self._acquisitions = []
        for acquisition in acquisitions:
            self.add(acquisition)
        self.walker = walker
        self.name_fmt = name_fmt
        self.iteration = 1

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
        """Add *acquisition* to the acquisition list and make it accessible as attribute, e.g.::

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
        tmp = first
        self._acquisitions[first_index] = second
        self._acquisitions[second_index] = tmp

    def get_acquisition(self, name):
        """
        Get acquisition by its *name*. In case there are more like it, the first
        one is returned.
        """
        for acq in self._acquisitions:
            if acq.name == name:
                return acq
        raise ExperimentError("Acquisition with name `{}' not found".format(name))

    def acquire(self):
        """
        Acquire data by running the acquisitions. This is the method which implements
        the data acquisition and should be overriden if more functionality is required,
        unlike :meth:`~.Experiment.run`.
        """
        for acq in self._acquisitions:
            acq()

    @async
    def run(self):
        """
        run()

        Compute the next iteration and run the :meth:`~.base.Experiment.acquire`.
        """
        # The data is not supposed to be overwritten, so find an iteration which
        # hasn't been used yet
        while self.walker.exists(self.name_fmt.format(self.iteration)):
            self.iteration += 1

        self.walker.descend(self.name_fmt.format(self.iteration))

        try:
            self.prepare()
            self.acquire()
            self.finish()
        finally:
            self.walker.ascend()
            self.iteration += 1


class ExperimentError(Exception):
    """Experiment-related exceptions."""
    pass
