'''
Created on Mar 3, 2013

@author: farago
'''
import uuid
from control.events.dispatcher import dispatcher

class ControlObject(object):
    """
    An object's id unique for the duration of a process in which the object
    resides. Python's id() function does not guarantee unique object ids
    in terms of process lifetime, only in terms of object lifetime.
    """
    def __init__(self):
        self._id = uuid.uuid4()

    @property
    def object_id(self):
        return self._id

    def subscribe(self, sender, event, callback):
        """Subscribe to an event.
        
        @param event: event
        @param callback: callback function
        
        """
        dispatcher.subscribe(sender, event, callback)

    def wait(self, senders_messages, timeout=None):
        """Wait for a particular event to happen (blocks execution).
        
        @param event: event
        @param timeout: timeout [s]
        
        """
        dispatcher.wait(senders_messages, timeout)