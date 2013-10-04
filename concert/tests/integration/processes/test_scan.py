from concert.quantities import q
from concert.tests import assert_almost_equal
from concert.devices.motors.dummy import Motor as DummyMotor
from concert.devices.cameras.dummy import Camera
from concert.processes import scan, ascan, dscan, scan_param_feedback
from concert.tests import slow
from concert.tests.base import ConcertTest


def compare_sequences(first_sequence, second_sequence, assertion):
    for x, y in zip(first_sequence, second_sequence):
        assertion(x, y)


class TestScan(ConcertTest):

    def setUp(self):
        super(TestScan, self).setUp()
        self.motor = DummyMotor()

    def handle_scan(self, parameters):
        self.positions.append(parameters[0].get().result())

    def test_ascan(self):
        self.positions = []

        ascan([(self.motor['position'], -2 * q.mm, 10 * q.mm)],
              n_intervals=4,
              handler=self.handle_scan)

        expected = [-2 * q.mm, 1 * q.mm, 4 * q.mm, 7 * q.mm, 10 * q.mm]
        compare_sequences(self.positions, expected, assert_almost_equal)

    def test_dscan(self):
        self.positions = []

        self.motor.position = 2 * q.mm
        dscan([(self.motor['position'], 2 * q.mm, 10 * q.mm)],
              n_intervals=4,
              handler=self.handle_scan)

        expected = [4 * q.mm, 6 * q.mm, 8 * q.mm, 10 * q.mm, 12 * q.mm]
        compare_sequences(self.positions, expected, assert_almost_equal)

    def test_process(self):
        def feedback():
            return self.motor.position

        x, y = scan(self.motor['position'], feedback,
                    1 * q.mm, 10 * q.mm, 10).result()
        compare_sequences(x, y, self.assertEqual)

    def test_scan_param_feedback(self):
        p = self.motor['position']
        x, y = scan_param_feedback(p, p, 1 * q.mm, 10 * q.mm, 10).result()
        compare_sequences(x, y, self.assertEqual)
