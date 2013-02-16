import unittest
import controller.scan

from controller.controller import LinearMotor


class FakeController(LinearMotor):
    def __init__(self):
        super(FakeController, self).__init__(0.0, 50.0, 0.0)

    def set_motor_position(self, param, value):
        print param, value


class TestLinearMotor(unittest.TestCase):
    def setUp(self):
        self.dummy = FakeController()

    def test_initial_position(self):
        self.assertEqual(self.dummy.position.value, 0)

    def test_mesh_scan(self):
        positions = []

        def callback(controllers):
            positions.append(controllers[0].position)

        controller.scan.mesh([self.dummy], 1.0, callback)
        self.assertEqual(len(positions), 50)
