from concert.devices.samplechangers.dummy import SampleChanger
from concert.tests import TestCase


class TestSampleChanger(TestCase):

    async def asyncSetUp(self):
        await super(TestSampleChanger, self).asyncSetUp()
        self.samplechanger = await SampleChanger()

    def test_set_sample(self):
        self.samplechanger.sample = None
        self.assertEqual(None, self.samplechanger.sample)
