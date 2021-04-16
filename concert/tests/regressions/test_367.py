from nose.plugins.attrib import attr
from concert.tests import TestCase
from concert.quantities import q


class TestIssue367(TestCase):
    @attr("skip-travis")
    def test_degree_conversion(self):
        from concert.devices.cameras.uca import Camera
        camera = Camera("mock")

        camera.degree_value = q.Quantity(5, q.celsius)
        self.assertEqual(camera.degree_value.magnitude, 5.0)

        val = camera.degree_value.magnitude
        camera.degree_value = camera.degree_value + 5 * q.delta_degC
        self.assertEqual(camera.degree_value.magnitude, val + 5)
