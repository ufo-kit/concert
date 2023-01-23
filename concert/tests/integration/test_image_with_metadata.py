import shutil
import tifffile
import os
import tempfile
import ast
import numpy as np
from concert.coroutines.base import background
from concert.base import check
from concert.quantities import q
from concert.experiments.addons import ImageWriter
from concert.experiments.synchrotron import Radiography
from concert.devices.shutters.dummy import Shutter
from concert.devices.motors.dummy import LinearMotor
from concert.storage import DirectoryWalker
from concert.helpers import ImageWithMetadata
from concert.devices.cameras.dummy import Camera as DummyCamera
from concert.tests import TestCase


class Camera(DummyCamera):
    """
    Dummy camera, that adds the frame index as integer, float, double and string to the images'
    metadata.
    """
    async def __ainit__(self, background=None, simulate=True):
        await super().__ainit__(background=background, simulate=simulate)
        self._frame_iterator = None

    @background
    @check(source='standby', target='recording')
    async def start_recording(self):
        self._frame_iterator = 0
        await super().start_recording()

    async def grab(self) -> ImageWithMetadata:
        img = await super().grab()
        img.metadata['index_int'] = int(self._frame_iterator)
        img.metadata['index_float'] = float(self._frame_iterator)
        img.metadata['index_string'] = str(self._frame_iterator)
        self._frame_iterator += 1
        return img


class TestMetadataExperiment(TestCase):
    async def asyncSetUp(self):
        self.camera = await Camera()
        self._data_dir = tempfile.mkdtemp()
        self.walker = DirectoryWalker(root=self._data_dir, bytes_per_file=1E12)
        flat_motor = await LinearMotor()
        shutter = await Shutter()
        self.exp = await Radiography(walker=self.walker,
                                     camera=self.camera,
                                     flat_motor=flat_motor,
                                     shutter=shutter,
                                     radio_position=0 * q.mm,
                                     flat_position=10 * q.mm,
                                     num_projections=100,
                                     num_flats=5,
                                     num_darks=5,
                                     separate_scans=False)
        self.writer = ImageWriter(self.exp.acquisitions, self.walker)

    def tearDown(self) -> None:
        shutil.rmtree(self._data_dir)

    async def test_metadata(self):
        """
        Runs a radiography. After wards the 'radios' are read and the tiff-files metadata is
        checked.
        """
        self.walker.descend("metadata")
        await self.exp.run()
        radio_images = tifffile.TiffReader(
            os.path.join(self.walker.current, "radios", "frame_000000.tif"))
        for i in range(await self.exp.get_num_projections()):
            self.assertEqual(ast.literal_eval(radio_images.pages[i].description)['index_int'],
                             int(i))
            self.assertEqual(ast.literal_eval(radio_images.pages[i].description)['index_float'],
                             float(i))
            self.assertEqual(ast.literal_eval(radio_images.pages[i].description)['index_string'],
                             str(i))
