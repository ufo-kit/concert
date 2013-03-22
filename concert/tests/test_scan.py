import time
import unittest
import logbook
import quantities as q
from concert.devices.motors.base import LinearCalibration
from concert.devices.motors.dummy import DummyMotor
from concert.processes.scan import ascan, dscan


class TestScan(unittest.TestCase):
    def setUp(self):
        self._motor = DummyMotor(LinearCalibration(1 / q.mm, 0 * q.mm))
        self.handler = logbook.TestHandler()
        self.handler.push_thread()

    def tearDown(self):
        self.handler.pop_thread()

    def test_ascan(self):
        self.positions = []

        def on_set_position(motor):
            self.positions.append(motor.get_position())

        self._motor.subscribe('position', on_set_position)
        ascan([(self._motor, -2 * q.mm, 10 * q.mm)], 4, blocking=True)
        time.sleep(0.05)

        self.assertEqual(len(self.positions), 5)
        self.assertEqual(self.positions[0], -2 * q.mm)
        self.assertEqual(self.positions[1], 1 * q.mm)
        self.assertEqual(self.positions[2], 4 * q.mm)
        self.assertEqual(self.positions[3], 7 * q.mm)
        self.assertEqual(self.positions[4], 10 * q.mm)

    def test_dscan(self):
        self.positions = []

        def on_set_position(motor):
            self.positions.append(motor.get_position())

        self._motor.set_position(2 * q.mm).wait()
        # FIXME: this would not be necessary, if we had a tighter callback
        # mechanism.
        time.sleep(0.05)

        self._motor.subscribe('position', on_set_position)
        dscan([(self._motor, 2 * q.mm, 10 * q.mm)], 4, blocking=True)
        time.sleep(0.05)

        self.assertEqual(len(self.positions), 5)
        self.assertEqual(self.positions[0], 4 * q.mm)
        self.assertEqual(self.positions[1], 6 * q.mm)
        self.assertEqual(self.positions[2], 8 * q.mm)
        self.assertEqual(self.positions[3], 10 * q.mm)
        self.assertEqual(self.positions[4], 12 * q.mm)
