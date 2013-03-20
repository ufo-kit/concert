import unittest
import logbook
import quantities as q
from concert.devices.axes.base import Axis, LinearCalibration
from concert.devices.axes.dummy import DummyAxis
from concert.processes.scan import ascan


class TestScan(unittest.TestCase):
    def setUp(self):
        self._axis = DummyAxis(LinearCalibration(1 / q.mm, 0 * q.mm))
        self.handler = logbook.TestHandler()
        self.handler.push_thread()

    def tearDown(self):
        self.handler.pop_thread()

    def test_ascan(self):
        self.positions = []

        def on_set_position(axis):
            self.positions.append(axis.get_position())

        self._axis.subscribe('position', on_set_position)
        ascan([(self._axis, -2 * q.mm, 10 * q.mm)], 4, True)

        self.assertEqual(len(self.positions), 5)
        self.assertEqual(self.positions[0], -2 * q.mm)
        self.assertEqual(self.positions[1], 1 * q.mm)
        self.assertEqual(self.positions[2], 4 * q.mm)
        self.assertEqual(self.positions[3], 7 * q.mm)
        self.assertEqual(self.positions[4], 10 * q.mm)


