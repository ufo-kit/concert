'''
Created on Mar 3, 2013

@author: farago
'''
from listener import MotionEventListener
import time
import numpy
import quantities as pq
from device.motion.axis.axis import DummyDiscreteAxis
from motion.axis.calibration import LinearCalibration


class StartStopListener(MotionEventListener):
    def on_start(self, event):
        print "%s: start." % (event.source)
    
    def on_stop(self, event):
        print "%s: stop." % (event.source)
        
    def on_limit_breach(self, event):
        print "%s: limit breach: %s." % (event.source, event.data)
        
        
if __name__ == '__main__': 
    da = DummyDiscreteAxis(LinearCalibration(pq.mm, 0*pq.mm), (-5,5))
    ssl = StartStopListener()
    for i in range(10):
        da.set_position(numpy.random.random()*pq.mm)
    time.sleep(1)