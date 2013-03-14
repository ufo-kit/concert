'''
Created on Mar 13, 2013

@author: farago
'''
from concert.devices.axes.base import LinearCalibration
import quantities as pq
import logging
from concert.processes.dummygradientmaximizer import DummyGradientMaximizer,\
    DummyGradientMaximizerState
from concert.devices.axes.ankatango import ANKATangoDiscreteAxis
from concert.connection import TangoConnection
from concert.devices.axes.dummy import DummyAxis
from concert.optimization.scalar import Maximizer
from concert.measures.dummygradient import DummyGradientMeasure




if __name__ == '__main__':
    logger = logging.getLogger("")
    logger.setLevel(logging.DEBUG)
    logging.basicConfig()
    
    axis = DummyAxis(LinearCalibration(1/pq.mm, 0*pq.mm))
    sharpness = Maximizer(1e-3)
    measue = DummyGradientMeasure(axis, 170.45*pq.mm)
    
    focuser = DummyGradientMaximizer(axis, 1.0*pq.mm, 1e-5,
                                     measue.get_gradient)
    focuser.focus(blocking=False)
    focuser.wait(DummyGradientMaximizerState.MAXIMUM_FOUND)
    print "Focus found at %s with gradient %g." % (str(axis.get_position()),
                                                   measue.get_gradient())