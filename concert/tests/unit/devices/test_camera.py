from datetime import datetime
import numpy as np
from concert.tests import TestCase
from concert.coroutines.base import coroutine
from concert.quantities import q
from concert.devices.cameras.dummy import Camera, BufferedCamera
from concert.devices.cameras.pco import Timestamp, TimestampError


class TestDummyCamera(TestCase):

    def setUp(self):
        super(TestDummyCamera, self).setUp()
        self.camera = Camera()

    def test_grab(self):
        frame = self.camera.grab()
        self.assertIsNotNone(frame)

    def test_trigger_mode(self):
        self.camera.trigger_mode = self.camera.trigger_modes.EXTERNAL
        self.assertEqual(self.camera.trigger_mode, 'EXTERNAL')

    def test_roi(self):
        self.roi_x0 = self.roi_y0 = self.roi_width = self.roi_height = 10 * q.dimensionless
        self.assertEqual(self.roi_x0, 10 * q.dimensionless)
        self.assertEqual(self.roi_y0, 10 * q.dimensionless)
        self.assertEqual(self.roi_width, 10 * q.dimensionless)
        self.assertEqual(self.roi_height, 10 * q.dimensionless)

    def test_has_sensor_dims(self):
        self.assertTrue(hasattr(self.camera, "sensor_pixel_width"))
        self.assertTrue(hasattr(self.camera, "sensor_pixel_height"))

    def test_buffered_camera(self):
        camera = BufferedCamera()
        for i, item in enumerate(camera.readout_buffer()):
            pass
        self.assertEqual(i, 2)

    def test_context_manager(self):
        camera = Camera()

        with camera.recording():
            self.assertEqual(camera.state, 'recording')
            f = camera.grab()

        self.assertEqual(camera.state, 'standby')

    def test_stream(self):
        @coroutine
        def check():
            while True:
                yield
                check.ok = True
                self.camera.stop_recording()
        check.ok = False

        self.camera.stream(check()).join()
        self.assertTrue(check.ok)


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
