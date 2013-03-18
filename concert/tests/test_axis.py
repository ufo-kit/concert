import unittest
import logbook
import quantities as q
from concert.tests import slow
from concert.devices.axes.base import LinearCalibration, Axis
from concert.devices.axes.dummy import DummyAxis, DummyContinuousAxis
from concert.events.dispatcher import wait


class TestDummyAxis(unittest.TestCase):
    def setUp(self):
        calibration = LinearCalibration(1 / q.mm, 0 * q.mm)
        self.axis = DummyAxis(calibration)
        self.handler = logbook.TestHandler()
        self.handler.push_thread()

    def tearDown(self):
        self.handler.pop_thread()

    def test_set_position_blocking(self):
        position = 1 * q.mm
        self.axis.set_position(position, True)
        new_position = self.axis.get_position()
        self.assertEqual(position, new_position)

    def test_set_position_nonblocking(self):
        position = 1 * q.mm
        e = self.axis.set_position(position, False)
        wait([e])
        self.assertEqual(position, self.axis.get_position())

    def test_set_positions_nonblocking(self):
        axis1 = DummyAxis(LinearCalibration(1 / q.mm, 0 * q.mm))

        position = 1 * q.mm
        position1 = 3 * q.mm

        event_1 = self.axis.set_position(position, False)
        event_2 = axis1.set_position(position1, False)
        wait([event_1, event_2])
        self.assertEqual(position, self.axis.get_position())
        self.assertEqual(position1, axis1.get_position())

    def test_move(self):
        position = 1 * q.mm
        delta = 0.5 * q.mm
        self.axis.set_position(position, True)
        self.axis.move(delta, True)
        self.assertEqual(position + delta, self.axis.get_position())


class TestContinuousDummyAxis(unittest.TestCase):
    def setUp(self):
        position_calibration = LinearCalibration(1 / q.mm, 0 * q.mm)
        velocity_calibration = LinearCalibration(1 / (q.mm / q.s),
                                                 0 * (q.mm / q.s))

        self.axis = DummyContinuousAxis(position_calibration,
                                        velocity_calibration)

        self.handler = logbook.TestHandler()
        self.handler.push_thread()

    def tearDown(self):
        self.handler.pop_thread()

    @slow
    def test_set_velocity_blocking(self):
        velocity = 1 * q.mm / q.s
        self.axis.set_velocity(velocity, True)
        new_velocity = self.axis.get_velocity()
        self.assertEqual(velocity, new_velocity)


class TestAxisCalibration(unittest.TestCase):
    def setUp(self):
        self.steps_per_mm = 10. / q.mm
        calibration = LinearCalibration(self.steps_per_mm, 0 * q.mm)

        class MockAxis(Axis):
            def __init__(self):
                super(MockAxis, self).__init__(calibration)

                self.position = 0 * q.dimensionless
                self._register('position',
                               self._get_position,
                               self._set_position,
                               q.m)

            def _stop_real(self):
                pass

            def _set_position(self, position):
                self.position = position

            def _get_position(self):
                return self.position

        self.axis = MockAxis()
        self.handler = logbook.TestHandler()
        self.handler.push_thread()

    def tearDown(self):
        self.handler.pop_thread()

    def test_set_position(self):
        position = 100 * q.mm
        steps = position * self.steps_per_mm

        self.axis.set_position(position, True)
        self.assertEqual(self.axis.position, steps)
        self.assertEqual(self.axis.get_position(), position)
