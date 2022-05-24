from concert.tests import TestCase
from concert.quantities import q


class TestIssue367(TestCase):

    async def test_degree_conversion(self):
        try:
            from concert.devices.cameras.uca import Camera
            camera = await Camera("mock")
        except Exception as err:
            self.skipTest(str(err))

        await camera.set_degree_value(q.Quantity(5, q.celsius))
        self.assertEqual((await camera.get_degree_value()).magnitude, 5.0)

        val = (await camera.get_degree_value()).magnitude
        await camera.set_degree_value(await camera.get_degree_value() + 5 * q.delta_degC)
        self.assertEqual((await camera.get_degree_value()).magnitude, val + 5)
