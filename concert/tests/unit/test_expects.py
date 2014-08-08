from concert.tests import TestCase
from concert.processes import focus, align_rotation_axis, find_beam
from concert.devices.motors.dummy import LinearMotor, RotationMotor
from concert.devices.cameras.dummy import Camera
from concert.quantities import q


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

    def test_align_rotation_axis_function(self):
        align_rotation_axis(
            self.camera, self.rotation_motor, x_motor=self.linear_motor)

    def test_find_beam_func_arguments_type_error(self):
        with self.assertRaises(TypeError):
            find_beam(self.camera, self.rotation_motor, self.linear_motor)

    def test_find_beam_function_arguments(self):
        find_beam(self.camera, self.linear_motor, self.linear_motor2,
                  [1, 1] * q.um, [1, 1] * q.um, [1, 1] * q.um)
