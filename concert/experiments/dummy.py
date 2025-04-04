"""Dummy experiments."""

import logging
import numpy as np
from concert.coroutines.base import run_in_executor
from concert.experiments.base import Acquisition, Experiment, local, remote
from concert.progressbar import wrap_iterable


LOG = logging.getLogger(__name__)


class ImagingExperiment(Experiment):

    """
    A typical imaging experiment which consists of acquiring dark, flat and radiographic images, in
    this case zeros or random data.

    .. py:attribute:: num_darks

        Number of dark images (no beam, just dark current)

    .. py:attribute:: num_flats

        Number of flat images (beam present, sample not)

    .. py:attribute:: num_radios

        Number of radiographic images

    .. py:attribute:: camera

        Camera to use for generating images

    .. py:attribute:: shape

        Shape of the generated images (H x W) (default: 1024 x 1024) if *camera* is not specified

    .. py:attribute:: random

        'off': use zeros
        'single': one random repeated in every iteration
        'multi': every iteration generates new random image

    .. py:attribute:: dtype

        Data type of the generated images (default: unsigned short)
    """

    async def __ainit__(self, num_darks, num_flats, num_radios, camera=None, shape=(1024, 1024),
                        walker=None, random=False, dtype=np.ushort, separate_scans=True,
                        name_fmt='scan_{:>04}'):
        self.num_darks = num_darks
        self.num_flats = num_flats
        self.num_radios = num_radios
        self.camera = camera
        self.shape = shape
        if random not in ['off', 'single', 'multi']:
            raise ValueError("random must be one of 'off', 'single', 'multi'")
        self.random = random
        self.dtype = dtype
        darks = await Acquisition('darks', self.take_darks)
        flats = await Acquisition('flats', self.take_flats)
        radios = await Acquisition('radios', self.take_radios)
        await super().__ainit__(
            [darks, flats, radios],
            walker=walker,
            separate_scans=separate_scans,
            name_fmt=name_fmt
        )

    async def _produce_images(self, num, mean=128, std=10):
        if self.camera:
            async with self.camera.recording():
                for i in wrap_iterable(list(range(num))):
                    yield await self.camera.grab()

        else:
            def make_random_image():
                return np.random.normal(mean, std, size=self.shape).astype(self.dtype)

            def make_const_image():
                return (np.ones(self.shape) * mean).astype(self.dtype)

            if self.random == 'off':
                image = await run_in_executor(make_const_image)
            elif self.random == 'single':
                image = await run_in_executor(make_random_image)

            for i in wrap_iterable(list(range(num))):
                if self.random == 'multi':
                    image = await run_in_executor(make_random_image)
                yield image

    @local
    async def take_darks(self):
        async for image in self._produce_images(self.num_darks):
            yield image

    @local
    async def take_flats(self):
        async for image in self._produce_images(self.num_flats):
            yield image

    @local
    async def take_radios(self):
        async for image in self._produce_images(self.num_radios):
            yield image


class ImagingFileExperiment(Experiment):

    """
    A typical imaging experiment which consists of acquiring dark, flat and radiographic images, in
    this case located on a disk.

    .. py:attribute:: camera

       A :class:`~concert.devices.cameras.dummy.FileCamera` object

    .. py:attribute:: num_darks

        Number of dark images (no beam, just dark current)

    .. py:attribute:: num_flats

        Number of flat images (beam present, sample not)

    .. py:attribute:: num_radios

        Number of radiographic images

    .. py:attribute:: darks_pattern

        Dark images file name pattern

    .. py:attribute:: flats_pattern

        Flat images file name pattern

    .. py:attribute:: radios_pattern

        Projection images file name pattern

    """

    async def __ainit__(self, camera, num_darks, num_flats, num_radios, darks_pattern='darks',
                        flats_pattern='flats', radios_pattern='projections', walker=None,
                        separate_scans=True, name_fmt='scan_{:>04}'):
        self.darks_pattern = darks_pattern
        self.flats_pattern = flats_pattern
        self.radios_pattern = radios_pattern
        self.num_darks = num_darks
        self.num_flats = num_flats
        self.num_radios = num_radios
        self.camera = camera
        darks = await Acquisition('darks', self.take_darks)
        flats = await Acquisition('flats', self.take_flats)
        radios = await Acquisition('radios', self.take_radios)
        await super().__ainit__(
            [darks, flats, radios],
            walker=walker,
            separate_scans=separate_scans,
            name_fmt=name_fmt
        )

    async def _produce_images(self, pattern, num):
        await self.camera.set_pattern(pattern)
        async with self.camera.recording():
            for i in wrap_iterable(list(range(num))):
                yield await self.camera.grab()

    @local
    async def take_darks(self):
        async for image in self._produce_images(self.darks_pattern, self.num_darks):
            yield image

    @local
    async def take_flats(self):
        async for image in self._produce_images(self.flats_pattern, self.num_flats):
            yield image

    @local
    async def take_radios(self):
        async for image in self._produce_images(self.radios_pattern, self.num_radios):
            yield image


class RemoteFileImagingExperiment(Experiment):

    """
    Uses a client camera and instead of yielding frames just tells the remote file camera to send
    them via network.

    .. py:attribute:: camera

       A :class:`~concert.devices.cameras.dummy.FileCamera` object

    .. py:attribute:: num_darks

        Number of dark images (no beam, just dark current)

    .. py:attribute:: num_flats

        Number of flat images (beam present, sample not)

    .. py:attribute:: num_radios

        Number of radiographic images

    .. py:attribute:: darks_pattern

        Dark images file name pattern

    .. py:attribute:: flats_pattern

        Flat images file name pattern

    .. py:attribute:: radios_pattern

        Projection images file name pattern

    """

    async def __ainit__(self, camera, num_darks, num_flats, num_radios, darks_pattern='darks',
                        flats_pattern='flats', radios_pattern='projections', walker=None,
                        separate_scans=True, name_fmt='scan_{:>04}'):
        self.darks_pattern = darks_pattern
        self.flats_pattern = flats_pattern
        self.radios_pattern = radios_pattern
        self.num_darks = num_darks
        self.num_flats = num_flats
        self.num_radios = num_radios
        self.camera = camera
        darks = await Acquisition('darks', self.take_darks, producer=camera)
        flats = await Acquisition('flats', self.take_flats, producer=camera)
        radios = await Acquisition('radios', self.take_radios, producer=camera)
        await super().__ainit__(
            [darks, flats, radios],
            walker=walker,
            separate_scans=separate_scans,
            name_fmt=name_fmt
        )

    async def _produce_images(self, pattern, number):
        await self.camera.set_pattern(pattern)
        async with self.camera.recording():
            await self.camera.grab_send(number)

        return number

    @remote
    def take_darks(self):
        return self._produce_images(self.darks_pattern, self.num_darks)

    @remote
    def take_flats(self):
        return self._produce_images(self.flats_pattern, self.num_flats)

    @remote
    def take_radios(self):
        return self._produce_images(self.radios_pattern, self.num_radios)
