'''
Created on Mar 13, 2013

@author: farago
'''
import unittest
from concert.devices.axes.dummy import DummyAxis
from concert.devices.axes.base import LinearCalibration
import quantities as pq
from concert.processes.dummygradientmaximizer import DummyGradientMaximizer,\
    DummyGradientMaximizerState
from concert.measures.dummygradient import DummyGradientMeasure


class TestDummyFocusing(unittest.TestCase):
    def setUp(self):
        self._axis = DummyAxis(LinearCalibration(1/pq.mm, 0*pq.mm))
        self._gradient_feedback = DummyGradientMeasure(self._axis, 18.75*pq.mm)
        self._focuser = DummyGradientMaximizer(self._axis, 1.0*pq.mm, 1e-3,
                                         self._gradient_feedback.get_gradient)
        self._position_eps = 1e-1*pq.mm
        self._gradient_cmp_eps = 1e-1
        
    def _check_gradient(self, cmp_gradient):
        gradient = self._gradient_feedback.get_gradient()
        self.assertTrue(cmp_gradient - self._gradient_cmp_eps\
            <= gradient <= cmp_gradient +\
            self._gradient_cmp_eps, "Gradient: %.8f differs " % (gradient) +\
            "more than by epsilon: %g" % (self._gradient_cmp_eps) +\
            " from the given gradient: %.8f." %\
            (self._gradient_feedback.maximum_gradient))
        
    def _check_position(self, cmp_position):
        self.assertTrue(cmp_position -
                self._position_eps <= self._axis.get_position() <=\
                cmp_position + self._position_eps, "Axis position: %s " %\
                (str(self._axis.get_position())) +\
                "differs more than by epsilon: %g " % (self._position_eps) +\
                "from the given position: %s" % (str(cmp_position)))
        
    def test_maximum_in_limits(self):
        self._focuser.focus()
        self._focuser.wait(DummyGradientMaximizerState.MAXIMUM_FOUND)
        self._check_position(self._gradient_feedback.max_gradient_position)
        self._check_gradient(self._gradient_feedback.maximum_gradient)
        
    def test_huge_step_in_limits(self):
        focuser = DummyGradientMaximizer(self._axis, 1000*pq.mm, 1e-3,
                                         self._gradient_feedback.get_gradient)
        focuser.focus()
        focuser.wait(DummyGradientMaximizerState.MAXIMUM_FOUND)
        self._check_position(self._gradient_feedback.max_gradient_position)
        self._check_gradient(self._gradient_feedback.maximum_gradient)
        
    def test_maximum_out_of_limits_right(self):
        self._gradient_feedback.max_gradient_position = \
                    (self._axis._hard_limits[1]+50)*pq.mm
        self._focuser.focus()
        self._focuser.wait(DummyGradientMaximizerState.MAXIMUM_FOUND)
        self._check_gradient(self._gradient_feedback.get_gradient())
        self._check_position(self._axis._hard_limits[1]*pq.mm)
    
    def test_maximum_out_of_limits_left(self):    
        self._gradient_feedback.max_gradient_position = \
                            (self._axis._hard_limits[0]-50)*pq.mm
        self._focuser.focus()
        self._focuser.wait(DummyGradientMaximizerState.MAXIMUM_FOUND)
        self._check_gradient(self._gradient_feedback.get_gradient())
        self._check_position(self._axis._hard_limits[0]*pq.mm)
        
    def test_huge_step_out_of_limits_right(self):
        self._gradient_feedback.max_gradient_position = \
                    (self._axis._hard_limits[1]+50)*pq.mm
        focuser = DummyGradientMaximizer(self._axis, 1000*pq.mm, 1e-3,
                                         self._gradient_feedback.get_gradient)
        focuser.focus()
        focuser.wait(DummyGradientMaximizerState.MAXIMUM_FOUND)
        self._check_gradient(self._gradient_feedback.get_gradient())
        self._check_position(self._axis._hard_limits[1]*pq.mm)
    
    def test_huge_step_out_of_limits_left(self):    
        self._gradient_feedback.max_gradient_position = \
                            (self._axis._hard_limits[0]-50)*pq.mm
        focuser = DummyGradientMaximizer(self._axis, 1000*pq.mm, 1e-3,
                                         self._gradient_feedback.get_gradient)
        focuser.focus()
        focuser.wait(DummyGradientMaximizerState.MAXIMUM_FOUND)
        self._check_gradient(self._gradient_feedback.get_gradient())
        self._check_position(self._axis._hard_limits[0]*pq.mm)