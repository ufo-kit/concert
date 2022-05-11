from concert.tests import TestCase
from nose.plugins.attrib import attr


class TestIssue400(TestCase):

    @attr("skip-ci")
    async def test_multiple_cameras(self):
        from concert.devices.cameras.uca import Camera

        cam_a = await Camera('mock')
        cam_b = await Camera('file')
        cam_c = await Camera('mock')

        self.assertEqual("degree_value" in dir(cam_b), False)
        self.assertEqual("path" in dir(cam_c), False)
