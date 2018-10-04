"""Dummy experiments."""

import os
import numpy as np
from concert.experiments.base import Acquisition, Experiment
from concert.devices.cameras.dummy import FileCamera
from concert.progressbar import wrap_iterable


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

    .. py:attribute:: shape

        Shape of the generated images (H x W) (default: 1024 x 1024)

    .. py:attribute:: random

        If True, one random image is created and re-used, otherwise zeros

    .. py:attribute:: dtype

        Data type of the generated images (default: unsigned short)
    """

    def __init__(self, num_darks, num_flats, num_radios, shape=(1024, 1024), walker=None,
                 random=False, dtype=np.ushort, separate_scans=True, name_fmt='scan_{:>04}'):
        self.num_darks = num_darks
        self.num_flats = num_flats
        self.num_radios = num_radios
        self.shape = shape
        self.random = random
        self.dtype = dtype
        darks = Acquisition('darks', self.take_darks)
        flats = Acquisition('flats', self.take_flats)
        radios = Acquisition('radios', self.take_radios)
        super(ImagingExperiment, self).__init__([darks, flats, radios], walker=walker)

    def _produce_images(self, num):
        if self.random:
            image = np.random.normal(128., 10., size=self.shape)
        else:
            image = np.zeros(self.shape)

        image = image.astype(self.dtype)
        for i in wrap_iterable(range(num)):
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

    def __init__(self, directory, num_darks, num_flats, num_radios, darks_dir='darks',
                 flats_dir='flats', radios_dir='projections', roi_x0=None, roi_width=None,
                 roi_y0=None, roi_height=None, walker=None, separate_scans=True,
                 name_fmt='scan_{:>04}'):
        self.directory = directory
        self.num_darks = num_darks
        self.num_flats = num_flats
        self.num_radios = num_radios
        self.darks_dir = darks_dir
        self.flats_dir = flats_dir
        self.radios_dir = radios_dir
        self.roi_x0 = roi_x0
        self.roi_width = roi_width
        self.roi_y0 = roi_y0
        self.roi_height = roi_height
        darks = Acquisition('darks', self.take_darks)
        flats = Acquisition('flats', self.take_flats)
        radios = Acquisition('radios', self.take_radios)
        super(ImagingFileExperiment, self).__init__([darks, flats, radios], walker=walker)

    def _produce_images(self, subdirectory, num):
        camera = FileCamera(os.path.join(self.directory, subdirectory))
        if self.roi_x0 is not None:
            camera.roi_x0 = self.roi_x0
        if self.roi_width is not None:
            camera.roi_width = self.roi_width
        if self.roi_y0 is not None:
            camera.roi_y0 = self.roi_y0
        if self.roi_height is not None:
            camera.roi_height = self.roi_height

        for i in wrap_iterable(range(num)):
            yield camera.grab()

    def take_darks(self):
        return self._produce_images(self.darks_dir, self.num_darks)

    def take_flats(self):
        return self._produce_images(self.flats_dir, self.num_flats)

    def take_radios(self):
        return self._produce_images(self.radios_dir, self.num_radios)
