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
        """Subscribe to an *event* sent by *sender*.
        
        When *sender* sends the particular event, *callback* will be called with
        the event is the first argument.
        """
        dispatcher.subscribe(sender, event, callback)

    def wait(self, senders_messages, timeout=None):
        """Wait for a particular set of events to happen.
        
        :func:`wait` blocks until either all messages are delivered or timeout,
        given in seconds, has passed.
        """
        dispatcher.wait(senders_messages, timeout)
