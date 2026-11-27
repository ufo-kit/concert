"""Index-based traversal independent of device values."""

import unittest
from itertools import product
from math import prod

from concert.scanning import scan_index


class TestScanOrder(unittest.TestCase):
    def test_raster(self):
        self.assertEqual([scan_index((2, 3), n) for n in range(6)],
                         list(product(range(2), range(3))))

    def test_snake(self):
        self.assertEqual([scan_index((3, 3), n, 'snake') for n in range(9)],
                         [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0),
                          (2, 0), (2, 1), (2, 2)])

    def test_spiral(self):
        self.assertEqual([scan_index((3, 3), n, 'spiral') for n in range(9)],
                         [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1),
                          (2, 0), (1, 0), (1, 1)])

    def test_spiral_rectangles_and_singletons(self):
        for rows in range(1, 7):
            for columns in range(1, 7):
                self.check_coverage_and_adjacency((rows, columns), 'spiral')

    def test_snake_nd(self):
        for shape in [(5,), (1, 4), (2, 3, 4), (3, 2, 3), (2, 1, 2, 3), (1, 1, 1)]:
            self.check_coverage_and_adjacency(shape, 'snake')

    def check_coverage_and_adjacency(self, shape, order):
        indices = [scan_index(shape, n, order) for n in range(prod(shape))]
        self.assertEqual(set(indices), set(product(*(range(size) for size in shape))))
        self.assertEqual(len(indices), len(set(indices)))
        for first, second in zip(indices, indices[1:]):
            self.assertEqual(sum(abs(a - b) for a, b in zip(first, second)), 1)

    def test_validation(self):
        for shape, iteration, order in [((2,), 0, 'spiral'), ((2, 2, 2), 0, 'spiral'),
                                        ((2, 2), 0, 'invalid'), ((), 0, 'snake'),
                                        ((0, 1), 0, 'raster')]:
            with self.assertRaises(ValueError):
                scan_index(shape, iteration, order)
        for iteration in [-1, 4]:
            with self.assertRaises(IndexError):
                scan_index((2, 2), iteration)
