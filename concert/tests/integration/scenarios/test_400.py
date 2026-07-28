from concert.tests import TestCase
from concert.devices.cameras.uca import Camera


class TestIssue400(TestCase):
    async def test_multiple_cameras(self):
        cam_a = await Camera(name='mock')
        cam_b = await Camera(name='file')
        cam_c = await Camera(name='mock')

        self.assertEqual("degree_value" in dir(cam_b), False)
        self.assertEqual("path" in dir(cam_c), False)
