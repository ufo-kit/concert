from concert.tests import TestCase


class TestIssue400(TestCase):

    async def test_multiple_cameras(self):
        try:
            from concert.devices.cameras.uca import Camera

            cam_a = await Camera('mock')
            cam_b = await Camera('file')
            cam_c = await Camera('mock')
        except Exception as err:
            self.skipTest(str(err))

        self.assertEqual("degree_value" in dir(cam_b), False)
        self.assertEqual("path" in dir(cam_c), False)
