from itertools import product
import numpy as np
from concert.quantities import q
from concert.tests import assert_almost_equal, TestCase
from concert.devices.motors.dummy import LinearMotor
from concert.processes.common import scan, ascan, dscan


def compare_sequences(first_sequence, second_sequence, assertion):
    assert len(first_sequence) == len(second_sequence)
    for x, y in zip(first_sequence, second_sequence):
        assertion(x[0], y[0])
        assertion(x[1], y[1])


class TestScan(TestCase):

    async def asyncSetUp(self):
        await super(TestScan, self).asyncSetUp()
        self.motor = await LinearMotor()

    async def feedback(self):
        return 1 * q.dimensionless

    async def test_ascan(self):

        async def run(include_last=True):
            scanned = []
            async for pair in ascan(self.motor['position'], 0 * q.mm, 10 * q.mm,
                                    5 * q.mm, self.feedback, include_last=include_last):
                scanned.append(pair)
            return scanned

        expected = [(0 * q.mm, 1 * q.dimensionless), (5 * q.mm, 1 * q.dimensionless),
                    (10 * q.mm, 1 * q.dimensionless)]

        scanned = await run()
        compare_sequences(expected, scanned, assert_almost_equal)

        # Second scan, values must be same
        scanned = await run()
        compare_sequences(expected, scanned, assert_almost_equal)

        # Exclude last
        scanned = await run(include_last=False)
        compare_sequences(expected[:-1], scanned, assert_almost_equal)

    async def test_ascan_units(self):
        scanned = []
        expected = [(0 * q.mm, 1 * q.dimensionless), (50 * q.mm, 1 * q.dimensionless),
                    (100 * q.mm, 1 * q.dimensionless)]

        async for pair in ascan(self.motor['position'], 0 * q.mm, 10 * q.cm,
                                5 * q.cm, self.feedback):
            scanned.append(pair)

        compare_sequences(expected, scanned, assert_almost_equal)

    async def test_dscan(self):
        async def run(include_last=True):
            scanned = []
            async for pair in dscan(self.motor['position'], 10 * q.mm, 5 * q.mm, self.feedback,
                                    include_last=include_last):
                scanned.append(pair)
            return scanned

        scanned = await run()
        expected = [(0 * q.mm, 1 * q.dimensionless), (5 * q.mm, 1 * q.dimensionless),
                    (10 * q.mm, 1 * q.dimensionless)]
        compare_sequences(expected, scanned, assert_almost_equal)

        # Second scan, x values must be different
        scanned = await run()
        expected = [(10 * q.mm, 1 * q.dimensionless), (15 * q.mm, 1 * q.dimensionless),
                    (20 * q.mm, 1 * q.dimensionless)]
        compare_sequences(expected, scanned, assert_almost_equal)

        # Exclude last
        scanned = await run(include_last=False)
        expected = [(20 * q.mm, 1 * q.dimensionless), (25 * q.mm, 1 * q.dimensionless)]
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
            other = await LinearMotor()
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
