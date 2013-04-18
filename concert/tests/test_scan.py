import time
import unittest
import logbook
import quantities as q
from concert.devices.motors.base import LinearCalibration
from concert.devices.motors.dummy import Motor as DummyMotor
from concert.processes.scan import Scanner, ascan, dscan


def compare_sequences(first_sequence, second_sequence, assertion):
    for x, y in zip(first_sequence, second_sequence):
        assertion(x, y)


class TestScan(unittest.TestCase):
    def setUp(self):
        self.motor = DummyMotor()
        self.handler = logbook.TestHandler()
        self.handler.push_application()

    def tearDown(self):
        self.handler.pop_application()

    def handle_scan(self, parameters):
        self.positions.append(parameters[0].get().result())

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

    def test_process(self):
        def feedback():
            return self.motor.position

        scanner = Scanner(self.motor['position'], feedback)
        scanner.minimum = 1 * q.mm
        scanner.maximum = 10 * q.mm
        scanner.intervals = 10
        x, y = scanner.run().result()
        compare_sequences(x, y, self.assertEqual)
