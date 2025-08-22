import numpy as np
from concert.quantities import q, Quantity
from concert.devices.motors.dummy import ContinuousRotationMotor
from concert.imageprocessing import find_needle_tips, find_sphere_centers
from concert.processes.common import align_rotation_axis, ProcessError
from concert.processes.common import align_rotation_stage_ellipse_fit
from concert.processes.common import DeviceSoftLimits, AcquisitionParams, AlignmentParams
from concert.tests import slow, TestCase
from concert.tests.util.rotationaxis import SimulationCamera


class TestDummyAlignment_SVD(TestCase):

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.x_motor = await ContinuousRotationMotor()
        self.y_motor = await ContinuousRotationMotor()
        self.z_motor = await ContinuousRotationMotor()

        await self.x_motor.set_position(0 * q.deg)
        await self.y_motor.set_position(0 * q.deg)
        await self.z_motor.set_position(0 * q.deg)

        self.camera = await SimulationCamera(128, self.x_motor["position"],
                                             self.y_motor["position"],
                                             self.z_motor["position"])

        # Allow 1 px misalignment in y-direction.
        self.eps = np.arctan(2 / self.camera.rotation_radius) * q.rad

    async def align_check(self, x_angle, z_angle, has_x_motor=True, has_z_motor=True,
                          get_ellipse_points=find_needle_tips, get_ellipse_points_kwargs=None):
        """"Align and check the results."""
        await self.x_motor.set_position(z_angle)
        await self.z_motor.set_position(x_angle)

        x_motor = self.x_motor if has_x_motor else None
        z_motor = self.z_motor if has_z_motor else None

        await align_rotation_axis(self.camera, self.y_motor, x_motor=x_motor, z_motor=z_motor,
                                  get_ellipse_points=get_ellipse_points,
                                  get_ellipse_points_kwargs=get_ellipse_points_kwargs)

        # In our case the best perfectly aligned position is when both
        # motors are in 0.
        if has_x_motor:
            assert np.abs(await self.x_motor.get_position()) < self.eps
        if has_z_motor:
            assert np.abs(await self.z_motor.get_position()) < self.eps

    async def test_no_motor(self):
        with self.assertRaises(ProcessError):
            await align_rotation_axis(self.camera, self.y_motor, x_motor=None, z_motor=None)

    @slow
    async def test_out_of_fov(self):
        async def get_ones():
            return np.ones((self.camera.size,
                            self.camera.size))

        self.camera._grab_real = get_ones

        with self.assertRaises(ProcessError) as ctx:
            await align_rotation_axis(self.camera, self.y_motor,
                                      x_motor=self.x_motor,
                                      z_motor=self.z_motor)

        self.assertEqual("Error finding reference points", str(ctx.exception))

    @slow
    async def test_not_offcentered(self):
        self.camera.rotation_radius = 0
        with self.assertRaises(ValueError) as ctx:
            await align_rotation_axis(self.camera, self.y_motor,
                                      x_motor=self.x_motor,
                                      z_motor=self.z_motor)

        self.assertEqual("Sample off-centering too "
                         + "small, enlarge rotation radius.",
                         str(ctx.exception))

    @slow
    async def test_no_x_axis(self):
        """Test the case when there is no x-axis motor available."""
        await self.align_check(17 * q.deg, 11 * q.deg, has_z_motor=False)

    @slow
    async def test_no_z_axis(self):
        """Test the case when there is no x-axis motor available."""
        await self.align_check(17 * q.deg, 11 * q.deg, has_x_motor=False)

    @slow
    async def test_not_misaligned(self):
        "Perfectly aligned rotation axis."
        await self.align_check(0 * q.deg, 0 * q.deg)

    @slow
    async def test_only_x(self):
        """Only misaligned laterally."""
        await self.align_check(-17 * q.deg, 0 * q.deg)

    @slow
    async def test_only_z(self):
        """Only misaligned in the beam direction."""
        await self.align_check(0 * q.deg, 11 * q.deg)

    @slow
    async def test_huge_x(self):
        self.camera.scale = (3, 0.25, 3)
        await self.align_check(60 * q.deg, 11 * q.deg)

    @slow
    async def test_huge_z(self):
        self.camera.scale = (3, 0.25, 3)
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
    async def test_sphere(self):
        """Test sphere instead of needle."""
        self.camera.rotation_radius = self.camera.size / 2
        self.camera.scale = (.5, .5, .5)
        self.camera.y_position = 0
        self.eps = np.arctan(2 / self.camera.rotation_radius) * q.rad
        await self.align_check(-17 * q.deg, 11 * q.deg,
                               get_ellipse_points=find_sphere_centers)

    @slow
    async def test_kwargs(self):
        """Test keyword arguments passing. Use sphere simulation and segmentation for that."""
        self.camera.rotation_radius = self.camera.size / 2
        self.camera.scale = (.25, .25, .25)
        self.camera.y_position = 0
        self.eps = np.arctan(2 / self.camera.rotation_radius) * q.rad
        await self.align_check(-17 * q.deg, 11 * q.deg,
                               get_ellipse_points=find_sphere_centers,
                               get_ellipse_points_kwargs={'correlation_threshold': 0.9})


class TestDummyAlignment_EllipseModel(TestCase):

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.pitch_motor = await ContinuousRotationMotor()
        self.tomo_motor = await ContinuousRotationMotor()
        self.roll_motor = await ContinuousRotationMotor()
        await self.pitch_motor.set_position(0 * q.deg)
        await self.tomo_motor.set_position(0 * q.deg)
        await self.roll_motor.set_position(0 * q.deg)
        self.camera = await SimulationCamera(512, self.pitch_motor["position"],
                                             self.tomo_motor["position"],
                                             self.roll_motor["position"],
                                             needle_radius=45,
                                             scales=(1, 1, 1),
                                             y_position=0)
        # Allow 1 px misalignment in y-direction.
        self.eps = np.arctan(1 / self.camera.rotation_radius) * q.rad

    async def align_check(self, roll_angle: Quantity, pitch_angle: Quantity,
                          acq_params: AcquisitionParams = AcquisitionParams(),
                          align_params: AlignmentParams = AlignmentParams()) -> None:
        """"Align and check the results."""
        await self.roll_motor.set_position(roll_angle)
        await self.pitch_motor.set_position(pitch_angle)
        await align_rotation_stage_ellipse_fit(camera=self.camera,
                                               tomo_motor=self.tomo_motor,
                                               rot_motor_roll=self.roll_motor,
                                               rot_motor_pitch=self.pitch_motor,
                                               dev_limits=DeviceSoftLimits(),
                                               acq_params=acq_params,
                                               align_params=align_params)
        # In our case the best perfectly aligned position is when both
        # motors are in 0.
        assert np.abs(await self.pitch_motor.get_position()) < self.eps.to(q.deg)
        assert np.abs(await self.roll_motor.get_position()) < self.eps.to(q.deg)

    @slow
    async def test_not_offcentered(self):
        self.camera.rotation_radius = 0
        with self.assertRaises(ValueError) as ctx:
            await align_rotation_stage_ellipse_fit(camera=self.camera,
                                               tomo_motor=self.tomo_motor,
                                               rot_motor_roll=self.roll_motor,
                                               rot_motor_pitch=self.pitch_motor,
                                               dev_limits=DeviceSoftLimits(),
                                               acq_params=AcquisitionParams(sphere_radius=45),
                                               align_params=AlignmentParams())
        self.assertEqual("sample off-centering too small, enlarge rotation radius",
                         str(ctx.exception))

    @slow
    async def test_not_enough_data_points(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            await align_rotation_stage_ellipse_fit(camera=self.camera,
                                               tomo_motor=self.tomo_motor,
                                               rot_motor_roll=self.roll_motor,
                                               rot_motor_pitch=self.pitch_motor,
                                               dev_limits=DeviceSoftLimits(),
                                               acq_params=AcquisitionParams(sphere_radius=45),
                                               align_params=AlignmentParams(num_frames=3))
        self.assertEqual("under-determined system, at least 5 coordinate pairs are needed",
                         str(ctx.exception))

    @slow
    async def test_not_misaligned(self) -> None:
        "Perfectly aligned rotation axis."
        self.camera.rotation_radius = 200
        await self.align_check(roll_angle=0 * q.deg, pitch_angle=0 * q.deg,
                               align_params=AlignmentParams(align_metric=self.eps))

    @slow
    async def test_only_roll(self) -> None:
        """Only misaligned laterally."""
        self.camera.rotation_radius = 200
        await self.align_check(roll_angle=-17 * q.deg, pitch_angle=0 * q.deg,
                               align_params=AlignmentParams(align_metric=self.eps))

    @slow
    async def test_only_pitch(self) -> None:
        """Only misaligned in the beam direction."""
        await self.align_check(roll_angle=0 * q.deg, pitch_angle=11 * q.deg,
                               align_params=AlignmentParams(align_metric=self.eps))

    @slow
    async def test_huge_roll(self) -> None:
        await self.align_check(roll_angle=60 * q.deg, pitch_angle=11 * q.deg,
                               align_params=AlignmentParams(ceil_rel_move=2 * q.deg,
                                                            align_metric=self.eps))

    @slow
    async def test_huge_pitch(self) -> None:
        await self.align_check(roll_angle=11 * q.deg, pitch_angle=60 * q.deg,
                               align_params=AlignmentParams(ceil_rel_move=2 * q.deg,
                                                            align_metric=self.eps))

    @slow
    async def test_positive(self) -> None:
        await self.align_check(roll_angle=17 * q.deg, pitch_angle=11 * q.deg,
                               align_params=AlignmentParams(align_metric=self.eps,
                                                            ceil_rel_move=0.5 * q.deg))

    @slow
    async def test_negative_positive(self) -> None:
        await self.align_check(roll_angle=-17 * q.deg, pitch_angle=11 * q.deg,
                               align_params=AlignmentParams(align_metric=self.eps,
                                                            ceil_rel_move=0.5 * q.deg))
        
    @slow
    async def test_positive_negative(self) -> None:
        await self.align_check(roll_angle=17 * q.deg, pitch_angle=-11 * q.deg,
                               align_params=AlignmentParams(align_metric=self.eps,
                                                            ceil_rel_move=0.5 * q.deg))
    @slow
    async def test_negative(self) -> None:
        await self.align_check(roll_angle=-17 * q.deg, pitch_angle=-11 * q.deg,
                               align_params=AlignmentParams(align_metric=self.eps,
                                                            ceil_rel_move=0.5 * q.deg))
