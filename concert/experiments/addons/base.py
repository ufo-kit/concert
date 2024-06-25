"""Add-ons for acquisitions are standalone extensions which can be applied to them. They operate on
the acquired data, e.g. write images to disk, do tomographic reconstruction etc.
"""
import logging
from concert.base import AsyncObject
from concert.experiments.base import Consumer as AcquisitionConsumer


LOG = logging.getLogger(__name__)


class Addon(AsyncObject):

    """A base addon class. An addon can be attached, i.e. its functionality is applied to the
    specified *acquisitions* and detached.

    .. py:attribute:: acquisitions

    A list of :class:`~concert.experiments.base._Acquisition` objects. The addon attaches itself on
    construction.

    """

    async def __ainit__(self, experiment, acquisitions=None):
        self.experiment = experiment
        self._consumers = set([])
        if acquisitions is not None:
            await self.attach(acquisitions)
        else:
            await self.attach(experiment.acquisitions)

    async def attach(self, acquisitions):
        """Attach the addon to *acquisitions*."""
        unattached = set(acquisitions)
        for acq in acquisitions:
            for consumer in self._consumers:
                if acq.contains(consumer):
                    unattached.remove(acq)

        consumers = self._make_consumers(unattached)

        for acq, consumer in consumers.items():
            acq.add_consumer(consumer)
            self._consumers.add(consumer)

        await self._setup()

    async def detach(self, acquisitions):
        """Detach the addon from *acquisitions*."""
        for acq in acquisitions:
            to_remove = set([])
            for consumer in self._consumers:
                if acq.contains(consumer):
                    acq.remove_consumer(consumer)
                    to_remove.add(consumer)
            self._consumers -= to_remove

        await self._teardown()

    async def _setup(self):
        pass

    async def _teardown(self):
        pass

    def _make_consumers(self, acquisitions):
        """
        Create consumers and store them in a dictionary in form {acquisition: consumer, ...}. Do not
        add the consumers to acquisitions, that is taken care of in :meth:`attach`.
        """
        raise NotImplementedError


class Benchmarker(Addon):

    """An addon which counts the time of acquisition duration.

    .. py:attribute:: acquisitions

    a list of :class:`~concert.experiments.base.Acquisition` objects

    """

    async def __ainit__(self, experiment, acquisitions=None):
        await super().__ainit__(experiment=experiment, acquisitions=acquisitions)

    def _make_consumers(self, acquisitions):
        consumers = {}

        for acq in acquisitions:
            consumers[acq] = AcquisitionConsumer(
                self.start_timer,
                corofunc_args=(acq.name,)
            )

        return consumers

    async def start_timer(self, *args, **kwargs):
        raise NotImplementedError

    async def get_duration(self, acquisition):
        return await self._get_duration(acquisition.name)

    async def _get_duration(self, acquisition_name):
        raise NotImplementedError


class ImageWriter(Addon):

    """An addon which writes images to disk.

    .. py:attribute:: acquisitions

    a list of :class:`~concert.experiments.base.Acquisition` objects

    .. py:attribute:: walker

    A :class:`~concert.storage.Walker` instance
    """

    async def __ainit__(self, experiment, acquisitions=None):
        self.walker = experiment.walker
        await super().__ainit__(experiment=experiment, acquisitions=acquisitions)

    def _make_consumers(self, acquisitions):
        """Attach all acquisitions."""
        consumers = {}

        def prepare_wrapper(acquisition):
            async def prepare_and_write(*args):
                # Make sure the directory exists
                async with self.walker:
                    try:
                        await self.walker.descend(acquisition.name)
                        # Even though acquisition name is fixed, we don't know where in the file
                        # system we are, so this must be determined dynamically when the writing
                        # is about to start
                        if self.write_sequence.remote:
                            coro = self.write_sequence(acquisition.name)
                        else:
                            coro = self.write_sequence(acquisition.name, producer=args[0])
                    finally:
                        await self.walker.ascend()

                await coro

            return prepare_and_write

        for acq in acquisitions:
            consumers[acq] = AcquisitionConsumer(prepare_wrapper(acq))
            consumers[acq].corofunc.remote = self.write_sequence.remote

        return consumers

    def write_sequence(self, name, producer=None):
        """Organize image writing to subdirectory *name* and return a coroutine which does the
        actual writing. This function is called inside the context manager of the walker and thus
        guarantees the correct base path of the experiment but also `async with self.walker' is not
        allowed here because it would block.
        """
        raise NotImplementedError


class Consumer(Addon):

    """An addon which applies a specific coroutine-based consumer to acquisitions.

    .. py:attribute:: acquisitions

    a list of :class:`~concert.experiments.base.Acquisition` objects

    """

    async def __ainit__(self, consumer, experiment, acquisitions=None):
        await super().__ainit__(experiment=experiment, acquisitions=acquisitions)
        self._consumer = consumer

    def _make_consumers(self, acquisitions):
        consumers = {}

        for acq in acquisitions:
            consumers[acq] = AcquisitionConsumer(self.consume)

        return consumers

    def consume(self, *args, **kwargs):
        raise NotImplementedError


class LiveView(Addon):

    """An addon which applies a specific coroutine-based consumer to acquisitions.

    .. py:attribute:: acquisitions

    a list of :class:`~concert.experiments.base.Acquisition` objects

    """

    async def __ainit__(self, viewer, experiment, acquisitions=None):
        await super().__ainit__(experiment=experiment, acquisitions=acquisitions)
        self._viewer = viewer

    def _make_consumers(self, acquisitions):
        consumers = {}

        for acq in acquisitions:
            consumers[acq] = AcquisitionConsumer(self.consume)

        return consumers

    def consume(self, *args, **kwargs):
        raise NotImplementedError


class Accumulator(Addon):

    """An addon which accumulates data.

    .. py:attribute:: acquisitions

    a list of :class:`~concert.experiments.base.Acquisition` objects

    .. py:attribute:: shapes

    a list of shapes for different acquisitions

    .. py:attribute:: dtype

    the numpy data type
    """

    async def __ainit__(self, experiment, acquisitions=None, shapes=None, dtype=None):
        self._shapes = shapes
        self._dtype = dtype
        await super().__ainit__(experiment=experiment, acquisitions=acquisitions)

    def _make_consumers(self, acquisitions):
        shapes = (None,) * len(acquisitions) if self._shapes is None else self._shapes
        consumers = {}

        for i, acq in enumerate(acquisitions):
            consumers[acq] = AcquisitionConsumer(
                self.accumulate,
                corofunc_args=(acq.name,),
                corofunc_kwargs={'shape': shapes[i], 'dtype': self._dtype},
            )

        return consumers

    async def accumulate(self, *args, **kwargs):
        raise NotImplementedError

    async def get_items(self, acquisition):
        return await self._get_items(acquisition.name)

    async def _get_items(self, acquisition_name):
        raise NotImplementedError


class OnlineReconstruction(Addon):
    async def __ainit__(self, experiment, acquisitions=None, do_normalization=True,
                        average_normalization=True, walker=None, slice_directory='online-slices'):
        self._args = None
        self._do_normalization = do_normalization
        self.walker = walker
        self.slice_directory = slice_directory
        await super().__ainit__(experiment=experiment, acquisitions=acquisitions)

    def _make_consumers(self, acquisitions):
        consumers = {}

        if self._do_normalization:
            consumers[get_acq_by_name(acquisitions, 'darks')] = AcquisitionConsumer(
                self.update_darks
            )
            consumers[get_acq_by_name(acquisitions, 'flats')] = AcquisitionConsumer(
                self.update_flats
            )

        consumers[get_acq_by_name(acquisitions, 'radios')] = AcquisitionConsumer(
            self.reconstruct
        )

        return consumers

    async def get_slice(self, x=None, y=None, z=None):
        if [x, y, z].count(None) != 2:
            raise ValueError('Exactly one dimension must be specified')
        if x is not None:
            return await self._get_slice_x(x)
        if y is not None:
            return await self._get_slice_y(y)
        if z is not None:
            return await self._get_slice_z(z)

    @property
    def args(self):
        return self._args

    async def update_darks(self, *args, **kwargs):
        raise NotImplementedError

    async def update_flats(self, *args, **kwargs):
        raise NotImplementedError

    async def reconstruct(self, *args, **kwargs):
        raise NotImplementedError

    async def rereconstruct(self, slice_directory=None):
        """Rereconstruct cached projections and saved them to *slice_directory*, which is a full
        path.
        """
        raise NotImplementedError

    async def find_axis(self, region, z=0, store=False):
        """Find the rotation axis in the *region* as [from, to, step] and return it."""
        raise NotImplementedError

    async def get_volume(self):
        raise NotImplementedError

    async def _get_slice_x(self, index):
        raise NotImplementedError

    async def _get_slice_y(self, index):
        raise NotImplementedError

    async def _get_slice_z(self, index):
        raise NotImplementedError


class PhaseGratingSteppingFourierProcessing(Addon):
    """
    Addon for a grating interferometry stepping experiment to process the raw data. The order of the
    acquisitions can be changed.
    """

    async def __ainit__(self, experiment, output_directory="contrasts"):
        self._output_directory = output_directory
        self._dark_image = None
        self._reference_stepping = []
        self._object_stepping = []
        self.object_intensity = None
        self.object_phase = None
        self.object_visibility = None
        self.reference_intensity = None
        self.reference_phase = None
        self.reference_visibility = None
        self.intensity = None
        self.diff_phase = None
        self.visibility_contrast = None
        self.diff_phase_in_rad = None
        await super().__ainit__(experiment=experiment)

    def _make_consumers(self, acquisitions):
        consumers = {}
        consumers[get_acq_by_name(acquisitions, 'darks')] = AcquisitionConsumer(
            self.process_darks,
        )
        consumers[get_acq_by_name(acquisitions, 'reference_stepping')] = AcquisitionConsumer(
            self.process_stepping,
        )
        consumers[get_acq_by_name(acquisitions, 'object_stepping')] = AcquisitionConsumer(
            self.process_stepping,
        )

        return consumers

    async def get_object_intensity(self):
        raise NotImplementedError

    async def get_object_phase(self):
        raise NotImplementedError

    async def get_object_visibility(self):
        raise NotImplementedError

    async def get_reference_intensity(self):
        raise NotImplementedError

    async def get_reference_phase(self):
        raise NotImplementedError

    async def get_reference_visibility(self):
        raise NotImplementedError

    async def get_intensity(self):
        raise NotImplementedError

    async def get_diff_phase(self):
        raise NotImplementedError

    async def get_visibility_contrast(self):
        raise NotImplementedError

    async def get_diff_phase_in_rad(self):
        raise NotImplementedError


def get_acq_by_name(acquisitions, name):
    """Get acquisition by *name* from a list of *acquisitions*."""
    for acq in acquisitions:
        if acq.name == name:
            return acq

    raise AddonError(f"Acquisition `{name}' not found")


class AddonError(Exception):
    """Addon errors."""


class OnlineReconstructionError(Exception):
    pass
