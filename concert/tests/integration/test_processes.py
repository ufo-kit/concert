import numpy as np
from concert.tests import assert_almost_equal, TestCase
from concert.quantities import q
from concert.devices.cameras.dummy import Camera
from concert.devices.motors.dummy import LinearMotor, RotationMotor
from concert.devices.shutters.dummy import Shutter
from concert.processes.common import focus
from concert.processes.beamline import (acquire_dark, acquire_image_with_beam,
                                        determine_rotation_axis)
from concert.tests.util.focus import BlurringCamera, FOCUS_POSITION


class TestProcesses(TestCase):

    async def asyncSetUp(self):
        self.motor = await LinearMotor()
        self.camera = await Camera()
        self.shutter = await Shutter()

    async def test_focusing(self):
        await self.motor.set_position(40. * q.mm)
        camera = await BlurringCamera(self.motor)
        await focus(camera, self.motor)
        assert_almost_equal(await self.motor.get_position(), FOCUS_POSITION, 1e-2)

    async def test_acquire_dark(self):
        self.assertTrue(isinstance(await acquire_dark(self.camera, self.shutter), np.ndarray))

    async def test_acquire_image_with_beam(self):
        frame = await acquire_image_with_beam(self.camera, self.shutter, self.motor, 1 * q.mm)
        self.assertTrue(isinstance(frame, np.ndarray))
        self.assertEqual(await self.motor.get_position(), 1 * q.mm)

    async def test_determine_rotation_axis(self):
        rot_motor = await RotationMotor()
        axis = await determine_rotation_axis(self.camera, self.shutter, self.motor, rot_motor,
                                             1 * q.mm, 3 * q.mm)
        self.assertTrue(isinstance(axis, q.Quantity))
