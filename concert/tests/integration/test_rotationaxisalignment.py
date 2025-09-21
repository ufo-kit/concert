import numpy as np
from concert.quantities import q, Quantity
from concert.devices.motors.dummy import ContinuousRotationMotor
from concert.imageprocessing import find_needle_tips, find_sphere_centers
from concert.processes.common import align_rotation_axis, ProcessError
from concert.processes.common import align_rotation_stage_comparative
from concert.processes.common import AcquisitionDevices, AcquisitionParams
from concert.processes.common import AlignmentDevices, AlignmentParams
from concert.devices.cameras.dummy import TomographyStageCamera
from concert.tests import slow, TestCase
from concert.tests.util.rotationaxis import SimulationCamera


class TestDummyAlignment_EllipseFit(TestCase):

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


class TestDummyAlignment_Comparative(TestCase):

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.width = 2560
        self.height = 2160
        self.camera = await TomographyStageCamera(shape=(self.height, self.width))
        self.tomo = self.camera.stage.tomo_motor
        self.lamino = self.camera.stage.lamino_motor # rot_motor_pitch
        self.roll = self.camera.stage.roll_motor # rot_motor_roll
        self.align_ortho = self.camera.stage.orthogonal_motor_above # align_motor_obd
        self.align_par = self.camera.stage.parallel_motor_above # align_motor_pbd
        self.stage_ortho = self.camera.stage.orthogonal_motor_below # flat_motor
        self.stage_vert = self.camera.stage.vertical_motor_below # z_motor
        self.acq_devices = AcquisitionDevices(
            camera=self.camera,
            shutter=None,
            tomo_motor=self.tomo,
            flat_motor=self.stage_ortho,
            z_motor=self.stage_vert)
        self.acq_params = AcquisitionParams(height=self.height, width=self.width)
        self.align_devices = AlignmentDevices(
            rot_motor_pitch=self.lamino,
            rot_motor_roll=self.roll,
            align_motor_pbd=self.align_par,
            align_motor_obd=self.align_ortho)
        self.align_params = AlignmentParams()

    @slow
    async def test_align_pitch_roll(self) -> None:
        await self.stage_ortho.set_position(0.8 * q.mm)
        await self.align_ortho.set_position(0.5 * q.mm)
        await self.align_par.set_position(-0.4 * q.mm)
        await self.lamino.set_position(3 * q.deg)
        await self.roll.set_position(0.25 * q.deg)
        await align_rotation_stage_comparative(
            acq_devices=self.acq_devices, acq_params=self.acq_params,
            align_devices=self.align_devices, align_params=self.align_params,
            pixel_size_um=1 * q.um, use_threshold=False)
        self.assertTrue(await self.roll.get_position() < self.acq_params.align_metric)
        self.assertTrue(await self.lamino.get_position() < self.acq_params.align_metric)
