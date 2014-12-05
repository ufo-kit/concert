from itertools import product
import numpy as np
from concert.quantities import q
from concert.tests import assert_almost_equal, TestCase
from concert.devices.motors.dummy import LinearMotor
from concert.processes.common import scan, ascan, dscan, scan_param_feedback
from concert.async import resolve
from concert.helpers import Region


def compare_sequences(first_sequence, second_sequence, assertion):
    for x, y in zip(first_sequence, second_sequence):
        assertion(x, y)


class TestScan(TestCase):

    def setUp(self):
        super(TestScan, self).setUp()
        self.motor = LinearMotor()

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

        values = np.linspace(1, 10, 12) * q.mm
        param_range = Region(self.motor['position'], values)

        x, y = zip(*list(resolve(scan(feedback, param_range))))
        compare_sequences(x, y, self.assertEqual)

    def test_scan_param_feedback(self):
        p = self.motor['position']
        values = np.linspace(1, 10, 10) * q.mm
        scan_param = Region(p, values)

        x, y = zip(*resolve(scan_param_feedback(scan_param, p)))
        compare_sequences(x, y, self.assertEqual)

    def test_multiscan(self):
        """A 2D scan."""
        other = LinearMotor()
        values_0 = np.linspace(0, 10, 2) * q.mm
        values_1 = np.linspace(5, 10, 3) * q.mm
        range_0 = Region(self.motor['position'], values_0)
        range_1 = Region(other['position'], values_1)

        def feedback():
            return self.motor.position, other.position

        gen = resolve(scan(feedback, [range_0, range_1]))
        p_0, p_1, result = zip(*gen)
        result_x, result_y = zip(*result)

        first_expected = [0 * q.mm, 10 * q.mm]
        second_expected = [5 * q.mm, 7.5 * q.mm, 10 * q.mm]
        combined = list(product(first_expected, second_expected))
        p_0_exp, p_1_exp = zip(*combined)

        # test parameter values
        compare_sequences(p_0, p_0_exp, assert_almost_equal)
        compare_sequences(p_1, p_1_exp, assert_almost_equal)
        # feedback result is a tuple in this case, test both parts
        compare_sequences(result_x, p_0_exp, assert_almost_equal)
        compare_sequences(result_y, p_1_exp, assert_almost_equal)

    def test_callback(self):
        called = []
        motor = LinearMotor()
        values = np.linspace(0, 2, 3) * q.mm
        qrange = Region(motor['position'], values)

        def callback():
            called.append(motor.position.to(q.mm).magnitude)

        list(resolve(scan(lambda: None, qrange, callbacks={qrange: callback})))

        np.testing.assert_almost_equal(called, range(3))
