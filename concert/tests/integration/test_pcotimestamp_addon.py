import shutil
import tempfile
from datetime import datetime
import numpy as np

from concert.quantities import q
from concert.storage import DirectoryWalker
from concert.tests import TestCase
from concert.devices.cameras.pco import Timestamp
from concert.devices.cameras.pco import Camera as PCOCamera
from concert.devices.cameras.dummy import Camera as DummyCamera
from concert.helpers import ImageWithMetadata
from concert.experiments.synchrotron import Radiography
from concert.devices.motors.dummy import LinearMotor
from concert.devices.shutters.dummy import Shutter
from concert.experiments.addons import PCOTimestampCheck, ImageWriter, PCOTimestampCheckError
from concert.base import transition, Parameter, identity


def _int_to_int_array(number, length):
    array = np.zeros(length, dtype=np.uint16)
    for i in range(length):
        array[length-i-1] = number % 10
        number //= 10
    return array


def add_binary_timestamp(img, time: datetime, number):
    data = np.zeros(28, dtype=np.uint16)
    data[:8] = _int_to_int_array(number, 8)
    data[8:12] = _int_to_int_array(time.year, 4)
    data[12:14] = _int_to_int_array(time.month, 2)
    data[14:16] = _int_to_int_array(time.day, 2)
    data[16:18] = _int_to_int_array(time.hour, 2)
    data[18:20] = _int_to_int_array(time.minute, 2)
    data[20:22] = _int_to_int_array(time.second, 2)
    data[22:28] = _int_to_int_array(time.microsecond, 6)

    for i in range(14):
        d0 = data[i * 2] & 0xf
        d1 = data[i * 2 + 1] & 0xf
        img[0, i] = d0 << 4 | d1
    return img


class Camera(DummyCamera):
    random_timestamp_numbers = Parameter()

    async def __ainit__(self, background=None, simulate=True):
        self.number = 1
        self._random_numbers = None
        self._timestamp_enabled = True
        await DummyCamera.__ainit__(self, background, simulate)
        await self.set_random_timestamp_numbers(False)

    async def _get_random_timestamp_numbers(self):
        return self._random_numbers

    async def _set_random_timestamp_numbers(self, value: bool):
        self._random_numbers = bool(value)

    async def grab(self) -> ImageWithMetadata:
        return await PCOCamera.grab(self)

    @transition(target='recording')
    async def _record_real(self):
        self.number = 1

    async def _grab_real(self):
        img = await super()._grab_real()
        time = datetime.now()
        if not await self.get_random_timestamp_numbers():
            img = add_binary_timestamp(img, time, self.number)
        else:
            img = add_binary_timestamp(img, time, np.random.randint(1, 1000))
        self.number += 1
        return img


PCOCamera._grab_real = Camera._grab_real


class TestPCOTimestampAddon(TestCase):
    async def asyncSetUp(self) -> None:
        self.camera = await Camera()
        self._data_dir = tempfile.mkdtemp()
        self.walker = DirectoryWalker(root=self._data_dir)
        flat_motor = await LinearMotor()
        shutter = await Shutter()
        self.exp = await Radiography(walker=self.walker,
                                     camera=self.camera,
                                     flat_motor=flat_motor,
                                     shutter=shutter,
                                     radio_position=0 * q.mm,
                                     flat_position=10*q.mm,
                                     num_projections=10,
                                     num_flats=5,
                                     num_darks=5)
        self.addon = PCOTimestampCheck(self.exp)
        self.writer = ImageWriter(self.exp.acquisitions, self.walker)

    def tearDown(self) -> None:
        shutil.rmtree(self._data_dir)

    async def test_int_to_int_array(self):
        self.assertTrue(np.all(_int_to_int_array(123456789, 9) == [1, 2, 3, 4, 5, 6, 7, 8, 9]))
        self.assertTrue(np.all(_int_to_int_array(1234, 9) == [0, 0, 0, 0, 0, 1, 2, 3, 4]))

    async def test_add_binary_timestamp(self):
        """
        Tests adding timestamp to image.
        """
        img = np.zeros((1, 14), dtype=np.uint16)
        time = datetime(2020, 1, 2, 3, 4, 5, 6)
        img = add_binary_timestamp(img, time, 789)
        timestamp = Timestamp(img)
        self.assertEqual(timestamp.number, 789)
        self.assertEqual(time, timestamp.time)

    async def test_camera(self):
        """
        Tests camera with timestamps.
        """
        await self.camera.set_random_timestamp_numbers(False)
        async with self.camera.recording():
            for i in range(10):
                img = await self.camera.grab()
                timestamp = Timestamp(img)
                self.assertEqual(timestamp.number, i + 1)

    async def run_test_addon(self):
        """
        Tests PCOTimestampCheck addon.
        """
        await self.camera.set_random_timestamp_numbers(False)
        await self.exp.run()
        self.assertFalse(self.addon.timestamp_incorrect)

        await self.camera.set_random_timestamp_numbers(True)
        with self.assertRaises(PCOTimestampCheckError):
            await self.exp.run()

        self.assertTrue(self.addon.timestamp_incorrect)

    async def test_addon_without_camera_convert(self):
        """
        Tests PCOTimestampCheck addon with camera.convert=identity
        """
        self.camera.convert = identity
        await self.run_test_addon()

    async def test_addon_with_camera_convert_views(self):
        """
        Tests PCOTimestampCheck addon with camera.convert="numpy-view"
        """
        self.camera.convert = np.fliplr
        await self.run_test_addon()

        def slice_image(x):
            return np.rot90(x)[7, :]
        self.camera.convert = slice_image
        await self.run_test_addon()

    async def test_addon_with_camera_convert_no_views(self):
        """
        Tests PCOTimestampCheck addon with camera.convert that creates a new array.
        """
        def delete_full_image(_):
            return np.zeros((100, 100))
        self.camera.convert = delete_full_image
        await self.run_test_addon()
