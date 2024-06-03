"""Dummy experiments."""

import logging
import os
import numpy as np
from concert.coroutines.base import run_in_executor
from concert.experiments.base import Acquisition, Experiment
from concert.devices.cameras.dummy import FileCamera
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
        darks = await Acquisition('darks', np.ndarray, self.take_darks)
        flats = await Acquisition('flats', np.ndarray, self.take_flats)
        radios = await Acquisition('radios', np.ndarray, self.take_radios)
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

    def take_darks(self):
        return self._produce_images(self.num_darks)

    def take_flats(self):
        return self._produce_images(self.num_flats)

    def take_radios(self):
        return self._produce_images(self.num_radios)


class ImagingFileExperiment(Experiment):

    """
    A typical imaging experiment which consists of acquiring dark, flat and radiographic images, in
    this case located on a disk.

    .. py:attribute:: directory

       Top directory with subdirectories containing the individual images

    .. py:attribute:: num_darks

        Number of dark images (no beam, just dark current)

    .. py:attribute:: num_flats

        Number of flat images (beam present, sample not)

    .. py:attribute:: num_radios

        Number of radiographic images

    .. py:attribute:: darks_dir

        Subdirectory name with dark images

    .. py:attribute:: flats_dir

        Subdirectory name with flat images

    .. py:attribute:: radio_dir

        Subdirectory name with radiographic images

    .. py:attribute:: roi_x0

        First read column

    .. py:attribute:: roi_width

        Number of read columns

    .. py:attribute:: roi_y0

        First read row

    .. py:attribute:: roi_height

        Number of read rows
    """

    async def __ainit__(self, directory, num_darks, num_flats, num_radios, darks_pattern='darks',
                        flats_pattern='flats', radios_pattern='projections', roi_x0=None,
                        roi_width=None, roi_y0=None, roi_height=None, walker=None,
                        separate_scans=True, name_fmt='scan_{:>04}', camera_class=FileCamera):
        self.directory = directory
        self.num_darks = num_darks
        self.num_flats = num_flats
        self.num_radios = num_radios
        self.darks_pattern = darks_pattern
        self.flats_pattern = flats_pattern
        self.radios_pattern = radios_pattern
        self.roi_x0 = roi_x0
        self.roi_width = roi_width
        self.roi_y0 = roi_y0
        self.roi_height = roi_height
        self._camera_class = camera_class
        darks = await Acquisition('darks', self._camera_class, self.take_darks)
        flats = await Acquisition('flats', self._camera_class, self.take_flats)
        radios = await Acquisition('radios', self._camera_class, self.take_radios)
        await super().__ainit__(
            [darks, flats, radios],
            walker=walker,
            separate_scans=separate_scans,
            name_fmt=name_fmt
        )

    async def _produce_images(self, pattern, num):
        camera = await self._camera_class(os.path.join(self.directory, pattern))
        if self.roi_x0 is not None:
            await camera.set_roi_x0(self.roi_x0)
        if self.roi_width is not None:
            await camera.set_roi_width(self.roi_width)
        if self.roi_y0 is not None:
            await camera.set_roi_y0(self.roi_y0)
        if self.roi_height is not None:
            await camera.set_roi_height(self.roi_height)

        async with camera.recording():
            for i in wrap_iterable(list(range(num))):
                yield await camera.grab()

    def take_darks(self):
        return self._produce_images(self.darks_pattern, self.num_darks)

    def take_flats(self):
        return self._produce_images(self.flats_pattern, self.num_flats)

    def take_radios(self):
        return self._produce_images(self.radios_pattern, self.num_radios)


class RemoteFileImagingExperiment(Experiment):

    """
    Uses a client camera and instead of yielding frames just tells the remote file camera to send
    them via network.
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
        self._camera = camera
        darks = await Acquisition('darks', camera, self.take_darks)
        flats = await Acquisition('flats', camera, self.take_flats)
        radios = await Acquisition('radios', camera, self.take_radios)
        await super().__ainit__(
            [darks, flats, radios],
            walker=walker,
            separate_scans=separate_scans,
            name_fmt=name_fmt
        )

    async def _produce_frames(self, pattern, number):
        await self._camera.set_pattern(pattern)
        async with self._camera.recording():
            await self._camera.grab_send(number)

        return number

    def take_darks(self):
        return self._produce_frames(self.darks_pattern, self.num_darks)

    def take_flats(self):
        return self._produce_frames(self.flats_pattern, self.num_flats)

    def take_radios(self):
        return self._produce_frames(self.radios_pattern, self.num_radios)
