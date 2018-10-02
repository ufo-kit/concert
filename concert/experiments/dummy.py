"""Dummy experiments."""

import os
from concert.experiments.base import Acquisition, Experiment
from concert.devices.cameras.dummy import FileCamera
from concert.progressbar import wrap_iterable


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
    """

    def __init__(self, directory, num_darks, num_flats, num_radios, darks_dir='darks',
                 flats_dir='flats', radios_dir='projections', walker=None, separate_scans=True,
                 name_fmt='scan_{:>04}'):
        self.directory = directory
        self.num_darks = num_darks
        self.num_flats = num_flats
        self.num_radios = num_radios
        self.darks_dir = darks_dir
        self.flats_dir = flats_dir
        self.radios_dir = radios_dir
        darks = Acquisition('darks', self.take_darks)
        flats = Acquisition('flats', self.take_flats)
        radios = Acquisition('radios', self.take_radios)
        super(ImagingFileExperiment, self).__init__([darks, flats, radios], walker=walker)

    def _produce_images(self, subdirectory, num):
        camera = FileCamera(os.path.join(self.directory, subdirectory))
        for i in wrap_iterable(range(num)):
            yield camera.grab()

    def take_darks(self):
        return self._produce_images(self.darks_dir, self.num_darks)

    def take_flats(self):
        return self._produce_images(self.flats_dir, self.num_flats)

    def take_radios(self):
        return self._produce_images(self.radios_dir, self.num_radios)
