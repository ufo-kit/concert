'''
Created on Mar 3, 2013

@author: farago
'''
from concert.events.dispatcher import dispatcher


class ConcertObject(object):
    """
    An object's id unique for the duration of a process in which the object
    resides. Python's id() function does not guarantee unique object ids
    in terms of process lifetime, only in terms of object lifetime.
    """
    def subscribe(self, events, callback):
        """Subscribe to a list of *events* composed of (sender, message)
        tuples. If a sender is None, the method will subscribe to a message
        coming from all senders.
        
        When *sender* sends the particular event, *callback* will be called with
        the event is the first argument.
        """
        dispatcher.subscribe(events, callback)

    def wait(self, events, timeout=None):
        """Wait for a particular list of *events* composed of (sender, message)
        tuples to happen. When *timeout* is given, the method will give every
        event *timeout* time to happen. Neither sender nor message can be None.
        
        :func:`wait` blocks until either all messages are delivered or timeout,
        given in seconds, has passed.
        """
        dispatcher.wait(events, timeout)