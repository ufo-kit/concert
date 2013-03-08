'''
Created on Mar 3, 2013

@author: farago
'''
import time
import numpy
import quantities as pq
from control.devices.motion.axes.dummyaxis import DummyAxis
from control.devices.motion.axes.calibration import LinearCalibration
from control.connections.dummyconnection import DummyConnection
from listener import AxisStateListener


class DummyListener(AxisStateListener):
    def on_moving(self, source):
        print "%s: moving." % (source)

    def on_standby(self, source):
        print "%s: standby." % (source)

    def on_position_limit(self, source):
        print "%s: position limit breach." % (source)


if __name__ == '__main__':
    da = DummyAxis(DummyConnection(),
           LinearCalibration(1 / pq.mm, 0 * pq.mm), (-5, 5))
    ssl = DummyListener()
    for i in range(5):
        da.set_position(numpy.random.random() * pq.mm)
    time.sleep(1)
