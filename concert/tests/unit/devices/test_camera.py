from datetime import datetime
import numpy as np
from concert.tests import TestCase
from concert.quantities import q
from concert.devices.cameras.dummy import Camera, BufferedCamera
from concert.devices.cameras.pco import Timestamp, TimestampError


class TestDummyCamera(TestCase):

    async def asyncSetUp(self):
        await super(TestDummyCamera, self).asyncSetUp()
        self.background = np.ones((256, 256), dtype=np.uint16)
        self.camera = await Camera(background=self.background)

    async def test_grab(self):
        frame = await self.camera.grab()
        self.assertIsNotNone(frame)

    def test_trigger_source(self):
        self.camera.trigger_source = self.camera.trigger_sources.EXTERNAL
        self.assertEqual(self.camera.trigger_source, 'EXTERNAL')

    def test_roi(self):
        self.roi_x0 = self.roi_y0 = self.roi_width = self.roi_height = 10 * q.dimensionless
        self.assertEqual(self.roi_x0, 10 * q.dimensionless)
        self.assertEqual(self.roi_y0, 10 * q.dimensionless)
        self.assertEqual(self.roi_width, 10 * q.dimensionless)
        self.assertEqual(self.roi_height, 10 * q.dimensionless)

    def test_has_sensor_dims(self):
        self.assertTrue(hasattr(self.camera, "sensor_pixel_width"))
        self.assertTrue(hasattr(self.camera, "sensor_pixel_height"))

    async def test_buffered_camera(self):
        camera = await BufferedCamera()
        i = 0
        async for item in camera.readout_buffer():
            i += 1
        self.assertEqual(i, 3)

    async def test_context_manager(self):
        camera = await Camera()

        async with camera.recording():
            self.assertEqual(await camera.get_state(), 'recording')
            await camera.grab()

        self.assertEqual(await camera.get_state(), 'standby')

    async def test_stream(self):
        async def check(producer):
            async for image in producer:
                check.ok = True
                await self.camera.stop_recording()
                break
        check.ok = False

        await check(self.camera.stream())
        self.assertTrue(check.ok)

    async def test_grab_convert(self):
        async def grab():
            return np.mgrid[:5, :5][1]

        self.camera._grab_real = grab
        self.camera.convert = np.fliplr
        image = await self.camera.grab()
        np.testing.assert_equal(image, (await grab())[:, ::-1])

    async def test_simulate(self):
        self.assertTrue(np.any(self.background - await self.camera.grab()))
        camera = await Camera(background=self.background, simulate=False)
        np.testing.assert_equal(self.background, await camera.grab())


class TestPCOTimeStamp(TestCase):
    def test_valid(self):
        image = np.empty((1, 14), dtype=np.uint16)
        image[0] = np.array([0, 0, 0, 1, 32, 20, 8, 1, 20, 80, 69, 147, 40, 9], dtype=np.uint16)
        stamp = Timestamp(image)
        date_time = datetime(2014, 8, 1, 14, 50, 45, 932809)

        self.assertEqual(stamp.time, date_time)
        self.assertEqual(stamp.number, 1)

    def test_bad_input(self):
        image = np.empty((1,))
        with self.assertRaises(TypeError):
            Timestamp(image)

        image = np.empty((1,), dtype=np.uint16)
        with self.assertRaises(ValueError):
            Timestamp(image)

    def test_bad_data(self):
        image = np.empty((1, 14), dtype=np.uint16)
        image[0] = np.array([0, 0, 0, 1, 32, 20, 2**15, 1, 20, 80, 69, 147, 40, 9], dtype=np.uint16)
        with self.assertRaises(TimestampError):
            Timestamp(image)
