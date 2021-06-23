from itertools import product
import numpy as np
from concert.quantities import q
from concert.tests import assert_almost_equal, TestCase
from concert.devices.motors.dummy import LinearMotor
from concert.processes.common import scan, ascan, dscan


def compare_sequences(first_sequence, second_sequence, assertion):
    for x, y in zip(first_sequence, second_sequence):
        assertion(x[0], y[0])
        assertion(x[1], y[1])


class TestScan(TestCase):

    def setUp(self):
        super(TestScan, self).setUp()
        self.motor = LinearMotor()

    async def feedback(self):
        return 1 * q.dimensionless

    async def test_ascan(self):

        async def run():
            scanned = []
            async for pair in ascan(self.motor['position'], 0 * q.mm, 10 * q.mm,
                                    5 * q.mm, self.feedback):
                scanned.append(pair)
            return scanned

        expected = [(0 * q.mm, 1 * q.dimensionless), (5 * q.mm, 1 * q.dimensionless)]

        scanned = await run()
        compare_sequences(expected, scanned, assert_almost_equal)

        # Second scan, values must be same
        scanned = await run()
        compare_sequences(expected, scanned, assert_almost_equal)

    async def test_dscan(self):
        async def run():
            scanned = []
            async for pair in dscan(self.motor['position'], 10 * q.mm, 5 * q.mm, self.feedback):
                scanned.append(pair)
            return scanned

        scanned = await run()
        expected = [(0 * q.mm, 1 * q.dimensionless), (5 * q.mm, 1 * q.dimensionless)]
        compare_sequences(expected, scanned, assert_almost_equal)

        # Second scan, x values must be different
        scanned = await run()
        expected = [(5 * q.mm, 1 * q.dimensionless), (10 * q.mm, 1 * q.dimensionless)]
        compare_sequences(expected, scanned, assert_almost_equal)

    async def test_scan(self):
        async def run():
            scanned = []
            async for pair in scan(self.motor['position'], np.arange(0, 10, 5) * q.mm,
                                   self.feedback):
                scanned.append(pair)
            return scanned

        scanned = await run()
        expected = [(0 * q.mm, 1 * q.dimensionless), (5 * q.mm, 1 * q.dimensionless)]
        compare_sequences(expected, scanned, assert_almost_equal)

    async def test_multiscan(self):
        """A 2D scan."""
        values_0 = np.arange(0, 10, 5) * q.mm
        values_1 = np.arange(20, 30, 5) * q.mm

        async def run():
            other = LinearMotor()
            scanned = []
            async for pair in scan((self.motor['position'], other['position']),
                                   (values_0, values_1),
                                   self.feedback):
                vec, res = pair
                scanned.append((vec[0], vec[1], res))
            return scanned

        scanned = await run()
        expected = list(product(values_0, values_1, [1 * q.dimensionless]))
        x, y, z = list(zip(*scanned))
        x_gt, y_gt, z_gt = list(zip(*expected))
        assert_almost_equal(x, x_gt)
        assert_almost_equal(y, y_gt)
        assert_almost_equal(z, z_gt)
