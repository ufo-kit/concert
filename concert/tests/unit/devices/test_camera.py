from concert.tests import TestCase
from concert.quantities import q
from concert.devices.cameras.dummy import Camera, BufferedCamera


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
