import numpy as np
from concert import optimization
from concert.base import HardLimitError
from concert.quantities import q
from concert.tests import slow, assert_almost_equal, TestCase
from concert.measures import DummyGradientMeasure
from concert.devices.motors.dummy import LinearMotor
from concert.optimization import optimize_parameter


class TestDummyFocusingWithSoftLimits(TestCase):

    async def asyncSetUp(self):
        await super(TestDummyFocusingWithSoftLimits, self).asyncSetUp()
        self.motor = await LinearMotor(position=50 * q.mm)
        await self.motor['position'].set_lower(25 * q.mm)
        await self.motor['position'].set_upper(75 * q.mm)
        self.halver_kwargs = {"initial_step": 10 * q.mm,
                              "max_iterations": 1000}

    @slow
    async def test_maximum_out_of_soft_limits_right(self):
        feedback = DummyGradientMeasure(self.motor['position'], 80 * q.mm, negative=True)
        await optimize_parameter(self.motor["position"], feedback,
                                 await self.motor.get_position(),
                                 optimization.halver,
                                 alg_kwargs=self.halver_kwargs)
        assert_almost_equal(await self.motor.get_position(), 75 * q.mm)

    @slow
    async def test_maximum_out_of_soft_limits_left(self):
        feedback = DummyGradientMeasure(self.motor['position'], 20 * q.mm, negative=True)
        await optimize_parameter(self.motor["position"], feedback,
                                 await self.motor.get_position(),
                                 optimization.halver,
                                 alg_kwargs=self.halver_kwargs)
        assert_almost_equal(await self.motor.get_position(), 25 * q.mm)


class TestDummyFocusing(TestCase):

    async def asyncSetUp(self):
        await super(TestDummyFocusing, self).asyncSetUp()
        self.motor = await LinearMotor(upper_hard_limit=2000 * q.mm, lower_hard_limit=-2000 * q.mm)
        await self.motor.set_position(0 * q.mm)
        self.feedback = DummyGradientMeasure(self.motor['position'],
                                             18.75 * q.mm, negative=True)
        self.epsilon = 1 * q.um
        self.position_eps = 1e-1 * q.mm
        self.gradient_cmp_eps = 1e-1
        self.halver_kwargs = {"initial_step": 10 * q.mm,
                              "max_iterations": 3000}

    async def run_optimization(self, initial_position=1 * q.mm):
        # Reset position of DummyMotor
        self.motor._position = 0 * q.mm
        await optimize_parameter(self.motor["position"], self.feedback,
                                 initial_position, optimization.halver,
                                 alg_kwargs=self.halver_kwargs)

    async def check(self, other):
        assert_almost_equal(await self.motor.get_position(), other, 0.1)

    @slow
    async def test_maximum_in_limits(self):
        # Disable Hard Limits of DummyMotor
        self.motor._upper_hard_limit = np.inf * q.mm
        self.motor._lower_hard_limit = -np.inf * q.mm

        await self.run_optimization()
        await self.check(self.feedback.max_position)

    @slow
    async def test_huge_step_in_limits(self):
        self.halver_kwargs["initial_step"] = 1000 * q.mm
        # Enable Hard Limits of DummyMotor
        self.motor._upper_hard_limit = 200 * q.mm
        self.motor._lower_hard_limit = -200 * q.mm

        with self.assertRaises(HardLimitError):
            await self.run_optimization()

    @slow
    async def test_maximum_out_of_limits_right(self):
        await self.motor['position'].set_upper(100 * q.mm)
        self.feedback.max_position = (await self.motor['position'].get_upper() + 50 * q.mm)

        with self.assertRaises(HardLimitError):
            await self.run_optimization()

    @slow
    async def test_maximum_out_of_limits_left(self):
        await self.motor['position'].set_lower(-100 * q.mm)
        self.feedback.max_position = (await self.motor['position'].get_lower() - 50 * q.mm)

        with self.assertRaises(HardLimitError):
            await self.run_optimization()

    @slow
    async def test_identical_gradients(self):
        # Some gradient level reached and then by moving to another position
        # the same level is reached again, but global gradient maximum
        # is not at this motor position.

        # Disable Hard Limits of DummyMotor
        self.motor._upper_hard_limit = np.inf * q.mm
        self.motor._lower_hard_limit = -np.inf * q.mm

        self.halver_kwargs["initial_step"] = 10 * q.mm
        await self.run_optimization(initial_position=-0.00001 * q.mm)
        await self.check(self.feedback.max_position)
