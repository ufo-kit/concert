"""Add-ons for acquisitions are standalone extensions which can be applied to them. They operate on
the acquired data, e.g. write images to disk, do tomographic reconstruction etc.
"""
import logging
from concert.coroutines.base import async_generate
from concert.coroutines.sinks import Accumulate


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
    """

    def __init__(self, acquisitions, walker):
        self.walker = walker
        self._writers = {}
        super(ImageWriter, self).__init__(acquisitions)

    def _attach(self):
        """Attach all acquisitions."""
        for acq in self.acquisitions:
            self._writers[acq] = self._write_sequence(acq)
            acq.consumers.append(self._writers[acq])

    def _detach(self):
        """Detach all acquisitions."""
        for acq in self.acquisitions:
            acq.consumers.remove(self._writers[acq])
            del self._writers[acq]

    def _write_sequence(self, acquisition):
        """Wrap the walker and write data."""
        async def wrapped_writer(producer):
            """Returned wrapper."""
            try:
                self.walker.descend(acquisition.name)
                await self.walker.write(producer)
            finally:
                self.walker.ascend()

        return wrapped_writer


class OnlineReconstruction(Addon):
    def __init__(self, experiment, reco_args, do_normalization=True,
                 average_normalization=True, walker=None, slice_directory='online-slices'):
        from concert.ext.ufo import GeneralBackprojectManager

        self.experiment = experiment
        self.manager = GeneralBackprojectManager(reco_args,
                                                 average_normalization=average_normalization)
        self.walker = walker
        self.slice_directory = slice_directory
        self._consumers = {}
        self._do_normalization = do_normalization
        super().__init__(experiment.acquisitions)

    async def _reconstruct(self, producer):
        await self.manager.backproject(producer)
        if self.walker:
            with self.walker.inside(self.slice_directory):
                producer = async_generate(self.manager.volume)
                await self.walker.write(producer, dsetname='slice_{:>04}.tif')

    def _attach(self):
        if self._do_normalization:
            self._consumers[self.experiment.darks] = self.manager.update_darks
            self._consumers[self.experiment.flats] = self.manager.update_flats
        self._consumers[self.experiment.radios] = self._reconstruct

        for acq, consumer in self._consumers.items():
            acq.consumers.append(consumer)

    def _detach(self):
        for acq, consumer in list(self._consumers.items()):
            acq.consumers.remove(consumer)


class AddonError(Exception):

    """Addon errors."""

    pass


class OnlineReconstructionError(Exception):
    pass
