'''
Created on Mar 3, 2013

@author: farago
'''
from listener import MotionEventListener
from device.motion.axis.dummy import DummyAxis
import time
import numpy


class StartStopListener(MotionEventListener):
    def on_start(self, event):
        print "%s: start." % (event.source)
    
    def on_stop(self, event):
        print "%s: stop." % (event.source)
        
    def on_position_changed(self, event):
        print "%s: position changed: %g." % (event.source, event.data)
        
if __name__ == '__main__':
    da = DummyAxis()
    ssl = StartStopListener()
    for i in range(10):
        da.set_position(numpy.random.random())
    time.sleep(1)