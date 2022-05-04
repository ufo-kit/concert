from concert.devices.attenuatorboxes.dummy import AttenuatorBox
from concert.tests import TestCase


class TestAttenuatorBox(TestCase):

    async def asyncSetUp(self):
        await super(TestAttenuatorBox, self).asyncSetUp()
        self.attenuatorBox = await AttenuatorBox()

    def test_set_attenuator(self):
        material0 = None
        self.attenuatorBox.attenuator = material0
        self.assertEqual(material0, self.attenuatorBox.attenuator)
        material1 = 'Al_1mm'
        self.attenuatorBox.attenuator = material1
        self.assertEqual(material1, self.attenuatorBox.attenuator)
