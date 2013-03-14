'''
Created on Mar 13, 2013

@author: farago
'''
import unittest
from concert.devices.axes.dummy import DummyAxis
from concert.devices.axes.base import LinearCalibration
import quantities as q
from concert.measures.dummygradient import DummyGradientMeasure
from concert.processes.focus import Focuser


class TestDummyFocusing(unittest.TestCase):
    def setUp(self):
        self._axis = DummyAxis(LinearCalibration(1/q.mm, 0*q.mm))
        self._gradient_feedback = DummyGradientMeasure(self._axis, 18.75*q.mm)
        self._focuser = Focuser(self._axis, 1e-3,
                                self._gradient_feedback.get_gradient)
        self._position_eps = 1e-1*q.mm
        self._gradient_cmp_eps = 1e-1

    def _check_position(self, cmp_position):
        self.assertTrue(cmp_position -
                        self._position_eps <= self._axis.get_position() <=
                        cmp_position + self._position_eps,
                        "Axis position: %s " %
                        (str(self._axis.get_position())) +
                        "differs more than by epsilon: %g " %
                        (self._position_eps) +
                        "from the given position: %s" % (str(cmp_position)))

    def test_maximum_in_limits(self):
        self._focuser.focus(1*q.mm, True)
        self._check_position(self._gradient_feedback.max_gradient_position)

    def test_huge_step_in_limits(self):
        self._focuser.focus(1000*q.mm, True)
        self._check_position(self._gradient_feedback.max_gradient_position)

    def test_maximum_out_of_limits(self):
        # Right.
        self._gradient_feedback.max_gradient_position = \
            (self._axis._hard_limits[1]+50)*q.mm
        self._focuser.focus(1*q.mm, True)
        self._check_position(self._axis._hard_limits[1]*q.mm)

        # Left.
        self._gradient_feedback.max_gradient_position = \
            (self._axis._hard_limits[0]-50)*q.mm
        self._focuser.focus(1*q.mm, True)
        self._check_position(self._axis._hard_limits[0]*q.mm)

    def test_huge_step_out_of_limits(self):
        # Right.
        self._gradient_feedback.max_gradient_position = \
            (self._axis._hard_limits[1]+50)*q.mm
        focuser = Focuser(self._axis, 1e-3,
                          self._gradient_feedback.get_gradient)
        focuser.focus(1000*q.mm, True)
        self._check_position(self._axis._hard_limits[1]*q.mm)

        # Left.
        self._gradient_feedback.max_gradient_position = \
            (self._axis._hard_limits[0]-50)*q.mm
        focuser = Focuser(self._axis, 1e-3,
                          self._gradient_feedback.get_gradient)
        focuser.focus(1000*q.mm, True)
        self._check_position(self._axis._hard_limits[0]*q.mm)

    def test_identical_gradients(self):
        # Some gradient level reached and then by moving to another position
        # the same level is reached again, but global gradient maximum
        # is not at this motor position.
        self._axis.set_position(-0.00001*q.mm, True)
        self._focuser.focus(10*q.mm, True)
        self._check_position(self._gradient_feedback.max_gradient_position)
