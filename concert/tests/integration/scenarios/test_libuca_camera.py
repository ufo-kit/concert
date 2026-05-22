from concert.tests import TestCase
from concert.devices.cameras.uca import Camera


class TestLibUcaCamera(TestCase):
    async def test_instantiation(self):
        camera = await Camera('mock')
        self.assertIsNotNone(camera)
        self.assertEqual(await camera.get_name(), 'mock camera')

    async def test_identical_plugin_manager(self):
        camera1 = await Camera('mock')
        camera2 = await Camera('mock')
        self.assertEqual(camera1._manager, camera2._manager)
