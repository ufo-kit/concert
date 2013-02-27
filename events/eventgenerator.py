'''
Created on Feb 27, 2013

@author: farago
'''
import zmq
import time
import eventformat
import itertools
import logging

class EventGenerator(object):
    """Event generator. ZMQ used for transporting messages. """
    # A unique id generator.
    event_id = itertools.count().next
    
    def __init__(self, protocol, host, port, ctx=None):
        """Constructor.
        
        @param protocol: a ZMQ protocol name
        @param host: host name
        @param port: port number
        @param ctx: a ZMQ context
        
        """
        self._logger = logging.getLogger(__name__ + "." +\
                                         self.__class__.__name__)
        if ctx is None:
            self._ctx = zmq.Context()
        else:
            # For inproc communication.
            self._ctx = ctx
        self._socket = self._ctx.socket(zmq.PUB)
        self._address = protocol+"://"+host
        if port is not None:
            self._address += ":"+str(port)
        self._socket.bind(self._address)
        self._logger.debug("Address: " + self._address)
        # Sleep a little to establish the connection properly.
        time.sleep(1)
        
    def _pack_message(self, event_type, source_id, data=""):
        msg = str(event_type)
        msg += eventformat.separator + str(source_id)
        msg += eventformat.separator + str(EventGenerator.event_id())
        msg += eventformat.separator + str(data)
        
        return msg
        
    def fire(self, event_type_id, source_id, data=""):
        """Fire an event.
        
        @param event_type_id: event type id (distibguishes different events)
        @param source_id: event generator id (distinguishes different devices)
        @param data: a string
        
        """
        msg = self._pack_message(event_type_id, source_id, data)
        self._logger.debug("Sending message (first KB): %s" % (msg[:1024]))
        self._socket.send(msg)