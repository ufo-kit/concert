import time
import unittest
import logbook
import quantities as q
from concert.devices.motors.base import LinearCalibration
from concert.devices.motors.dummy import DummyMotor
from concert.processes.scan import ascan, dscan


def compare_sequences(first_sequence, second_sequence, assertion):
    for x, y in zip(first_sequence, second_sequence):
        assertion(x, y)


class TestScan(unittest.TestCase):
    def setUp(self):
        self.motor = DummyMotor(LinearCalibration(1 / q.mm, 0 * q.mm))
        self.handler = logbook.TestHandler()
        self.handler.push_thread()

    def tearDown(self):
        self.handler.pop_thread()

    def handle_scan(self, parameters):
        self.positions.append(parameters[0].get())

    def test_ascan(self):
        self.positions = []

        ascan([(self.motor['position'], -2 * q.mm, 10 * q.mm)],
              n_intervals=4,
              handler=self.handle_scan)

        expected = [-2 * q.mm, 1 * q.mm, 4 * q.mm, 7 * q.mm, 10 * q.mm]
        compare_sequences(self.positions, expected, self.assertAlmostEqual)

    def test_dscan(self):
        self.positions = []

        self.motor.position = 2 * q.mm
        dscan([(self.motor['position'], 2 * q.mm, 10 * q.mm)],
              n_intervals=4,
              handler=self.handle_scan)

        expected = [4 * q.mm, 6 * q.mm, 8 * q.mm, 10 * q.mm, 12 * q.mm]
        compare_sequences(self.positions, expected, self.assertAlmostEqual)
