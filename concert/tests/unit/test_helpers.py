import time
from concert.tests import TestCase, suppressed_logging
from concert.quantities import q
from concert.helpers import measure
from concert.processes.common import focus, align_rotation_axis, ProcessError
from concert.devices.motors.dummy import LinearMotor, RotationMotor
from concert.devices.cameras.dummy import Camera


class TestExpects(TestCase):

    def setUp(self):
        self.seq = list(range(10))
        self.camera = Camera()
        self.linear_motor = LinearMotor()
        self.linear_motor2 = LinearMotor()
        self.rotation_motor = RotationMotor()

    async def test_focus_func_arguments_type_error(self):
        with self.assertRaises(TypeError):
            await focus(self.camera, self.rotation_motor)

    async def test_focus_function_arguments(self):
        await focus(self.camera, self.linear_motor)

    async def test_align_rotation_axis_func_type_error(self):
        with self.assertRaises(TypeError):
            await align_rotation_axis(self.camera, self.linear_motor).result()
        with self.assertRaises(TypeError):
            await align_rotation_axis(
                self.camera, self.rotation_motor, self.linear_motor).result()
        with self.assertRaises(TypeError):
            await align_rotation_axis(
                self.camera,
                self.rotation_motor,
                self.rotation_motor,
                self.rotation_motor,
                num_frames=[
                    10,
                    20]).result()
        with self.assertRaises(TypeError):
            await align_rotation_axis(
                self.camera,
                self.rotation_motor,
                self.rotation_motor,
                self.rotation_motor,
                num_frames=10 *
                q.mm).result()

    async def test_align_rotation_axis_function(self):
        with self.assertRaises(ProcessError):
            # Dummy camera, so no tips in noise
            await align_rotation_axis(self.camera, self.rotation_motor, x_motor=self.rotation_motor)


@suppressed_logging
def test_measure_execution():
    @measure(return_result=True)
    def sleeping():
        time.sleep(0.001)
        return 123

    result, elapsed = sleeping()
    assert(result == 123)
    assert(elapsed > 0.001 * q.s)
