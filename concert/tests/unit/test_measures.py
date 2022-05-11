import numpy as np
from concert.coroutines.base import async_generate
from concert.quantities import q
from concert.devices.motors.dummy import ContinuousRotationMotor
from concert.imageprocessing import find_needle_tips
from concert.tests import slow, TestCase
from concert.tests.util.rotationaxis import SimulationCamera
from concert.processes.common import scan, align_rotation_axis
from concert.measures import rotation_axis


class TestRotationAxisMeasure(TestCase):

    async def asyncSetUp(self):
        await super(TestRotationAxisMeasure, self).asyncSetUp()
        self.x_motor = await ContinuousRotationMotor()
        self.y_motor = await ContinuousRotationMotor()
        self.z_motor = await ContinuousRotationMotor()

        # The bigger the image size, the more images we need to determine
        # the center correctly.
        self.image_source = await SimulationCamera(128, self.x_motor["position"],
                                                   self.y_motor["position"],
                                                   self.z_motor["position"])

        # Allow 1 px misalignment in y-direction.
        self.eps = np.arctan(2 / self.image_source.rotation_radius) * q.rad

    async def make_images(self, x_angle, z_angle, intervals=10):
        await self.x_motor.set_position(z_angle)
        await self.z_motor.set_position(x_angle)
        values = np.linspace(0, 2 * np.pi, intervals) * q.rad
        async for pair in scan(self.y_motor["position"], values, self.image_source.grab):
            yield pair[1]

    async def align_check(self, x_angle, z_angle):
        images = self.make_images(x_angle, z_angle)
        tips = await find_needle_tips(images)
        phi, psi = rotation_axis(tips)[:2]

        assert phi - x_angle < self.eps
        assert np.abs(psi) - np.abs(z_angle) < self.eps

    async def center_check(self, images):
        tips = await find_needle_tips(images)
        center = rotation_axis(tips)[2]

        assert np.abs(center[1] - self.image_source.ellipse_center[1]) < 2
        assert np.abs(center[0] - self.image_source.ellipse_center[0]) < 2

    @slow
    async def test_out_of_fov(self):
        images = np.random.normal(size=(10, self.image_source.size, self.image_source.size))
        with self.assertRaises(ValueError) as ctx:
            tips = await find_needle_tips(async_generate(images))
            rotation_axis(tips)

        self.assertEqual("No sample tip points found.", str(ctx.exception))

    @slow
    async def test_center_no_rotation(self):
        images = self.make_images(0 * q.deg, 0 * q.deg, intervals=15)
        await self.center_check(images)

    @slow
    async def test_center_only_x(self):
        self.image_source.scale = (3, 0.33, 3)
        images = self.make_images(17 * q.deg, 0 * q.deg, intervals=15)
        await self.center_check(images)

    @slow
    async def test_center_only_z(self):
        images = self.make_images(0 * q.deg, 11 * q.deg, intervals=15)
        await self.center_check(images)

    @slow
    async def test_center_positive(self):
        images = self.make_images(17 * q.deg, 11 * q.deg, intervals=15)
        await self.center_check(images)

    @slow
    async def test_center_negative_positive(self):
        images = self.make_images(-17 * q.deg, 11 * q.deg, intervals=15)
        await self.center_check(images)

    @slow
    async def test_center_positive_negative(self):
        images = self.make_images(17 * q.deg, -11 * q.deg, intervals=15)
        await self.center_check(images)

    @slow
    async def test_center_negative(self):
        images = self.make_images(-17 * q.deg, -11 * q.deg, intervals=15)
        await self.center_check(images)

    @slow
    async def test_only_x(self):
        """Only misaligned laterally."""
        await self.align_check(0 * q.deg, 0 * q.deg)

    @slow
    async def test_only_z(self):
        """Only misaligned in the beam direction."""
        await self.align_check(0 * q.deg, 11 * q.deg)

    @slow
    async def test_huge_x(self):
        self.image_source.scale = (3, 0.25, 3)
        await self.align_check(60 * q.deg, 11 * q.deg)

    @slow
    async def test_huge_z(self):
        self.image_source.scale = (3, 0.25, 3)
        await self.align_check(11 * q.deg, 60 * q.deg)

    @slow
    async def test_positive(self):
        await self.align_check(17 * q.deg, 11 * q.deg)

    @slow
    async def test_negative_positive(self):
        await self.align_check(-17 * q.deg, 11 * q.deg)

    @slow
    async def test_positive_negative(self):
        await self.align_check(17 * q.deg, -11 * q.deg)

    @slow
    async def test_negative(self):
        await self.align_check(-17 * q.deg, -11 * q.deg)

    @slow
    async def test_pitch_sgn(self):
        self.image_source.size = 512
        # Image acquisition inverts the contrast, so invert it here to get it right there
        await self.x_motor.set_position(10 * q.deg)
        await self.y_motor.set_position(0 * q.deg)
        await self.z_motor.set_position(0 * q.deg)
        eps = 0.1 * q.deg
        await align_rotation_axis(self.image_source, self.y_motor, x_motor=self.x_motor,
                                  z_motor=self.z_motor, initial_x_coeff=2 * q.dimensionless,
                                  metric_eps=eps)

        assert np.abs(await self.x_motor.get_position()) < eps
