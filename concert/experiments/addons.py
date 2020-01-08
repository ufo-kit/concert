"""Add-ons for acquisitions are standalone extensions which can be applied to them. They operate on
the acquired data, e.g. write images to disk, do tomographic reconstruction etc.
"""
import logging
import numpy as np
from concert.async import threaded
from concert.coroutines.base import broadcast, coroutine
from concert.coroutines.filters import queue
from concert.coroutines.sinks import Accumulate, Result


LOG = logging.getLogger(__name__)


class Addon(object):

    """A base addon class. An addon can be attached, i.e. its functionality is applied to the
    specified *acquisitions* and detached.

    .. py:attribute:: acquisitions

    A list of :class:`~concert.experiments.base.Acquisition` objects. The addon attaches itself on
    construction.

    """

    def __init__(self, acquisitions):
        self.acquisitions = acquisitions
        self._attached = False
        self.attach()

    def attach(self):
        """Attach the addon to all acquisitions."""
        if self._attached:
            LOG.debug('Cannot attach an already attached Addon')
        else:
            self._attach()
            self._attached = True

    def detach(self):
        """Detach the addon from all acquisitions."""
        if self._attached:
            self._detach()
            self._attached = False
        else:
            LOG.debug('Cannot detach an unattached Addon')

    def _attach(self):
        """Attach implementation."""
        raise NotImplementedError

    def _detach(self):
        """Detach implementation."""
        raise NotImplementedError


class Consumer(Addon):

    """An addon which applies a specific coroutine-based consumer to acquisitions.

    .. py:attribute:: acquisitions

    a list of :class:`~concert.experiments.base.Acquisition` objects

    .. py:attribute:: consumer

    A callable which returns a coroutine which processes the incoming data from acquisitions

    """

    def __init__(self, acquisitions, consumer):
        self.consumer = consumer
        super(Consumer, self).__init__(acquisitions)

    def _attach(self):
        """Attach all acquisitions."""
        for acq in self.acquisitions:
            acq.consumers.append(self.consumer)

    def _detach(self):
        """Detach all acquisitions."""
        for acq in self.acquisitions:
            acq.consumers.remove(self.consumer)


class Accumulator(Addon):

    """An addon which accumulates data.

    .. py:attribute:: acquisitions

    a list of :class:`~concert.experiments.base.Acquisition` objects

    .. py:attribute:: shapes

    a list of shapes for different acquisitions

    .. py:attribute:: dtype

    the numpy data type
    """

    def __init__(self, acquisitions, shapes=None, dtype=None):
        self._accumulators = {}
        self._shapes = shapes
        self._dtype = dtype
        self.items = {}
        super(Accumulator, self).__init__(acquisitions)

    def _attach(self):
        """Attach all acquisitions."""
        shapes = (None,) * len(self.acquisitions) if self._shapes is None else self._shapes

        for i, acq in enumerate(self.acquisitions):
            self._accumulators[acq] = Accumulate(shape=shapes[i], dtype=self._dtype)
            self.items[acq] = self._accumulators[acq].items
            acq.consumers.append(self._accumulators[acq])

    def _detach(self):
        """Detach all acquisitions."""
        self.items = {}
        for acq in self.acquisitions:
            acq.consumers.remove(self._accumulators[acq])

        self._accumulators = {}


class ImageWriter(Addon):

    """An addon which writes images to disk.

    .. py:attribute:: acquisitions

    a list of :class:`~concert.experiments.base.Acquisition` objects

    .. py:attribute:: walker

    A :class:`~concert.storage.Walker` instance

    .. py:attribute:: async

    If True write images asynchronously
    """

    def __init__(self, acquisitions, walker, async=True):
        self.walker = walker
        self._async = async
        self._writers = {}
        super(ImageWriter, self).__init__(acquisitions)

    def _attach(self):
        """Attach all acquisitions."""
        for acq in self.acquisitions:
            block = True if acq == self.acquisitions[-1] else False
            self._writers[acq] = self._write_sequence(acq, block)
            acq.consumers.append(self._writers[acq])

    def _detach(self):
        """Detach all acquisitions."""
        for acq in self.acquisitions:
            acq.consumers.remove(self._writers[acq])
            del self._writers[acq]

    def _write_sequence(self, acquisition, block):
        """Wrap the walker and write data."""
        def wrapped_writer():
            """Returned wrapper."""
            try:
                self.walker.descend(acquisition.name)
                coro = self.walker.write()
                if self._async:
                    coro = queue(coro, process_all=True, block=block, make_deepcopy=False)
                return coro
            finally:
                self.walker.ascend()

        return wrapped_writer


class OnlineReconstruction(Addon):
    def __init__(self, experiment, reco_args, process_normalization=False,
                 consumer=None, block=False, wait_for_projections=False,
                 walker=None, slice_directory='online-slices'):
        from multiprocessing.pool import ThreadPool
        from threading import Event
        from concert.ext.ufo import GeneralBackprojectManager

        self.experiment = experiment
        self.dark_result = Result()
        self.flat_result = Result()
        self.manager = GeneralBackprojectManager(reco_args)
        self.walker = walker
        self.slice_directory = slice_directory
        self.process_normalization = process_normalization
        self._pool = ThreadPool(processes=2)
        self._events = {'darks': Event(), 'flats': Event()}
        self.consumer = consumer
        self.block = block
        self.wait_for_projections = wait_for_projections
        self._consumers = {}
        super(OnlineReconstruction, self).__init__(experiment.acquisitions)

    @threaded
    def _average_images(self, queue, im_type):
        average = None
        i = 0
        while True:
            image = queue.get()
            if image is None:
                if im_type == 'darks':
                    self.manager.dark = average
                else:
                    self.manager.flat = average
                self._events[im_type].set()
                LOG.debug('%s pre-processing done', im_type)
                break

            if self.process_normalization:
                if average is None:
                    average = np.zeros_like(image, dtype=np.float32)
                average = (average * i + image) / (i + 1)
                i += 1
            else:
                average = image

    def _create_averaging(self, im_type):
        @coroutine
        def create_averaging_coro():
            try:
                import Queue as queue_module
            except ImportError:
                import queue as queue_module

            self._events[im_type].clear()
            queue = queue_module.Queue()
            self._average_images(queue, im_type)
            try:
                while True:
                    image = yield
                    queue.put(image)
            except GeneratorExit:
                queue.put(None)

        return create_averaging_coro

    def _reconstruct(self):
        if hasattr(self.experiment, 'darks') and hasattr(self.experiment, 'flats'):
            events = self._events.values() if self.process_normalization else None
        else:
            events = None
        consumers = []
        write_coro = None

        if self.consumer:
            consumers.append(self.consumer)
        if self.walker:
            self.walker.descend(self.slice_directory)
            write_coro = self.walker.write(dsetname='slice_{:>04}.tif')
            self.walker.ascend()
            consumers.append(write_coro)
        consumer = broadcast(*consumers) if consumers else None

        return self.manager(consumer=consumer, block=self.block, wait_for_events=events,
                            wait_for_projections=self.wait_for_projections)

    def _attach(self):
        if hasattr(self.experiment, 'darks') and hasattr(self.experiment, 'flats'):
            self._consumers[self.experiment.darks] = self._create_averaging('darks')
            self._consumers[self.experiment.flats] = self._create_averaging('flats')
        self._consumers[self.experiment.radios] = self._reconstruct

        for acq, consumer in self._consumers.items():
            acq.consumers.append(consumer)

    def _detach(self):
        for acq, consumer in self._consumers.items():
            acq.consumers.remove(consumer)


class AddonError(Exception):

    """Addon errors."""

    pass


class OnlineReconstructionError(Exception):
    pass
