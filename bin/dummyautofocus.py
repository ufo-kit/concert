'''
Created on Mar 13, 2013

@author: farago
'''
from concert.devices.axes.base import LinearCalibration
import quantities as q
import logging
from concert.devices.axes.dummy import DummyAxis
from concert.measures.dummygradient import DummyGradientMeasure
from concert.processes.focus import Focuser
from concert.optimization.scalar import Maximizer
from concert.events.service import wait


if __name__ == '__main__':
    logger = logging.getLogger("")
    logger.setLevel(logging.DEBUG)
    logging.basicConfig()

    axis = DummyAxis(LinearCalibration(1/q.mm, 0*q.mm))
    sharpness = Maximizer(1e-3)
    measue = DummyGradientMeasure(axis, 18.75*q.mm)

    focuser = Focuser(axis, 1e-5, measue.get_gradient)
    event = focuser.focus(10*q.mm, blocking=False)
    wait([event])
    print "Focus found at %s with gradient %g." % (str(axis.get_position()),
                                                   measue.get_gradient())
