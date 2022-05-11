from concert.devices.lenschangers.dummy import LensChanger
from concert.tests import TestCase


class TestLensChanger(TestCase):

    async def asyncSetUp(self):
        await super(TestLensChanger, self).asyncSetUp()
        self.lens_changer = await LensChanger()

    def test_set_objective(self):
        lens0 = "objective_10x"
        self.lens_changer.objective = lens0
        self.assertEqual(lens0, self.lens_changer.objective)
        lens1 = 'objective_5x'
        self.lens_changer.objective = lens1
        self.assertEqual(lens1, self.lens_changer.objective)
