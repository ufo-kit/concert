from concert.tests.base import ConcertTest
from concert.devices.cameras.dummy import FileCamera
from concert.quantities import q
from concert.devices.cameras.base import CameraError


class TestFileCamera(ConcertTest):

    def setUp(self):
        super(TestFileCamera, self).setUp()
        self.camera = FileCamera(".")
        self.camera.fps = 100 * q.count / q.s

    def test_start_stop_recording(self):
        self.assertRaises(CameraError, self.camera.grab)
        self.assertRaises(CameraError, self.camera.stop_recording)

        # This must pass without any problems
        self.camera.start_recording()
        self.camera.stop_recording()
        try:
            self.camera.grab()
        except KeyError:
            # There are probably no supported files in current directory
            pass

    def test_trigger_mode(self):
        self.camera.trigger_mode = FileCamera.TRIGGER_SOFTWARE

        # This must pass
        self.camera.start_recording()
        self.camera.trigger()
        self.camera.stop_recording()

        # No trigger, nothing grabbed
        self.camera.start_recording()
        self.camera.stop_recording()
        self.assertEqual(self.camera.grab(), None)

        # This must fail
        self.camera.trigger_mode = FileCamera.TRIGGER_AUTO
        self.camera.start_recording()
        self.assertRaises(CameraError, self.camera.trigger)
        self.camera.stop_recording()
