from concert.tests import TestCase
from concert.devices.detectors.dummy import Detector


class TestDetector(TestCase):
    def setUp(self):
        super(TestDetector, self).setUp()
        self.detector = Detector()

    def test_pixel_size(self):
        pixel_width = self.detector.camera.sensor_pixel_width
        pixel_height = self.detector.camera.sensor_pixel_height

        self.assertEqual(self.detector.pixel_width, pixel_width * self.detector.magnification)
        self.assertEqual(self.detector.pixel_height, pixel_height * self.detector.magnification)
