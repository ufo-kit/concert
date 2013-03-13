'''
Created on Mar 3, 2013

@author: farago
'''
from concert.events.dispatcher import dispatcher


class ConcertObject(object):
    """
    Base concert object which can send, subscribe to and wait for messages.
    """
    def send(self, message):
        """Send a *message* tied with this object."""
        dispatcher.send(self, message)
    
    def subscribe(self, message, callback):
        """Subscribe to a *message* from this object.
        
        *callback* will be called with this object as the first argument.
        """
        dispatcher.subscribe([(self, message)], callback)
        
    def unsubscribe(self, message, callback):
        """
        Unsubscribe from a *message*.
        
        *callback* is a function which is unsubscribed from a particular
        *message* coming from this object.
        """
        dispatcher.unsubscribe([(self, message)], callback)

    def wait(self, message, timeout=None):
        """Wait for a *message* from this object.

        When *timeout* is given, the method will give a *message* *timeout*
        time to happen.

        .. note::

            This method blocks until the message is delivered or
            the timeout has passed.
        """
        dispatcher.wait([(self, message)], timeout)
