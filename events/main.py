'''
Created on Feb 27, 2013

Simple check of the ZMQ-based events functionality.

@author: farago
'''
import zmq
from eventgenerator import EventGenerator
import eventtype
import time
from eventlistener import MotionEventListener
import numpy

class DummyMotionListener(MotionEventListener):
    """A simple implementation of MotionEventListener interface."""
    def on_start(self, source_id, event_id, data):
        print "Start of device %s." % (source_id)
        
    def on_stop(self, source_id, event_id, data):
        print "Stop of device %s." % (source_id)
        
    def on_state_changed(self, source_id, event_id, data):
        print "State changed to %s of device %s." % (data, source_id)
        
    def on_position_changed(self, source_id, event_id, data):
        print "Position changed to %s of device %s." % (data, source_id)
        
    def on_limit_reached(self, source_id, event_id, data):
        print "Limit reached by device %s." % (source_id)

def generate_events(motor_ids, event_generator):
    for i in range(10):
        event_generator.fire(eventtype.start, motor_ids[i % len(motor_ids)])
        time.sleep(0.2)
        event_generator.fire(eventtype.state_changed,
                             motor_ids[i % len(motor_ids)], "moving")
        time.sleep(0.5)
        event_generator.fire(eventtype.position_changed,
                             motor_ids[i % len(motor_ids)],
                             numpy.random.random())
        time.sleep(1)
        event_generator.fire(eventtype.limit_reached,
                             motor_ids[i % len(motor_ids)])
        time.sleep(0.5)
        event_generator.fire(eventtype.stop, motor_ids[i % len(motor_ids)])
        time.sleep(0.2)

if __name__ == '__main__':
    # Create context, necessary to share for ZMQ inproc communication.
    ctx = zmq.Context()
    
    # Motor name. Will be queried from HWManager. This is not an integer
    # for interoperability purposes with the current TANGO scheme.
    device_ids = ["motor1", "motor2", "motor3", "motor4", "motor5"]
    # Listen to only some of the all devices.
    listen_to_ids = ["motor1", "motor3"]
    
    
    # Intra-process communication.
    event_generator = EventGenerator("inproc", "whatever", None, ctx)
    motion_event_listener = DummyMotionListener("inproc", "whatever", None,
                                                        listen_to_ids, ctx)
#    # Inter-process communication.
#    event_generator = EventGenerator("ipc", "/tmp/0", None)
#    motion_event_listener = DummyMotionListener("ipc", "/tmp/0", None,
#                                                        listen_to_ids)
#    # TCP communication.
#    event_generator = EventGenerator("tcp", "*", 10000)
#    motion_event_listener = DummyMotionListener("tcp", "localhost", 10000,
#                                                        listen_to_ids)
    # See if the communication works.
    generate_events(device_ids, event_generator)