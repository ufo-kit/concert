import time
from concert.tests import TestCase, suppressed_logging
from concert.quantities import q
from concert.helpers import measure
from concert.measures import rotation_axis
from concert.processes.common import focus, align_rotation_axis, find_beam
from concert.devices.motors.dummy import LinearMotor, RotationMotor
from concert.devices.cameras.dummy import Camera


class TestExpects(TestCase):

    def setUp(self):
        self.seq = range(10)
        self.camera = Camera()
        self.linear_motor = LinearMotor()
        self.linear_motor2 = LinearMotor()
        self.rotation_motor = RotationMotor()

    def test_focus_func_arguments_type_error(self):
        with self.assertRaises(TypeError):
            focus(self.camera, self.rotation_motor)

    def test_focus_function_arguments(self):
        focus(self.camera, self.linear_motor)

    def test_align_rotation_axis_func_type_error(self):
        with self.assertRaises(TypeError):
            align_rotation_axis(self.camera, self.linear_motor).result()
        with self.assertRaises(TypeError):
            align_rotation_axis(
                self.camera, self.rotation_motor, self.linear_motor).result()
        with self.assertRaises(TypeError):
            align_rotation_axis(
                self.camera,
                self.rotation_motor,
                self.rotation_motor,
                self.rotation_motor,
                measure=rotation_axis,
                num_frames=[
                    10,
                    20]).result()
        with self.assertRaises(TypeError):
            align_rotation_axis(
                self.camera,
                self.rotation_motor,
                self.rotation_motor,
                self.rotation_motor,
                measure=rotation_axis,
                num_frames=10 *
                q.mm).result()
        with self.assertRaises(TypeError):
            align_rotation_axis(
                self.camera,
                self.rotation_motor,
                self.rotation_motor,
                self.rotation_motor,
                measure=rotation_axis,
                num_frames=10,
                absolute_eps=[
                    1,
                    2] * q.deg).result()

    def test_align_rotation_axis_function(self):
        align_rotation_axis(
            self.camera, self.rotation_motor, x_motor=self.rotation_motor)

    def test_find_beam_func_arguments_type_error(self):
        with self.assertRaises(TypeError):
            find_beam(self.camera, self.rotation_motor, self.linear_motor)
        with self.assertRaises(TypeError):
            find_beam(
                self.camera, self.linear_motor, self.linear_motor2, [1, 2])
        with self.assertRaises(TypeError):
            find_beam(
                self.camera, self.linear_motor, self.linear_motor2, [1, 2, 3] * q.um)
        with self.assertRaises(TypeError):
            find_beam(
                self.camera, self.linear_motor, self.linear_motor2, [1, 2] * q.deg)

    def test_find_beam_function_arguments(self):
        find_beam(self.camera, self.linear_motor, self.linear_motor2,
                  [1, 1] * q.um, [1, 1] * q.um, [1, 1] * q.um)


@suppressed_logging
def test_measure_execution():
    @measure(return_result=True)
    def sleeping():
        time.sleep(0.001)
        return 123

    result, elapsed = sleeping()
    assert(result == 123)
    assert(elapsed > 0.001 * q.s)
    assert(elapsed < 0.010 * q.s)
