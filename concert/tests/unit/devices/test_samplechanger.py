from concert.devices.samplechangers.dummy import SampleChanger
from concert.tests import TestCase


class TestSampleChanger(TestCase):

    def setUp(self):
        super(TestSampleChanger, self).setUp()
        self.samplechanger = SampleChanger()

    def test_set_sample(self):
        self.samplechanger.sample = None
        self.assertEqual(None, self.samplechanger.sample)
