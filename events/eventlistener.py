'''
Created on Feb 27, 2013

@author: farago
'''
import zmq
import eventformat
import eventtype
from threading import Thread
import logging
from event import Event
from abc import ABCMeta, abstractmethod


class _EventListener(object):
    """Interface for listening to events. Must be inherited from. Uses ZMQ
    for transporting messages. A listener can listen to various events from
    various sources. The protocol follows publisher-subscriber pattern. This
    interface is private to this module and should not be implemented from
    outside. There are follow-up specific interfaces which are supposed to
    be implemented outside this module.
    """
    __metaclass__ = ABCMeta
    
    def __init__(self, protocol, host, port, event_types,
                                                sources=[""],  ctx=None):
        """Constructor.
        
        @param protocol: a ZMQ protocol name
        @param host: host name
        @param port: port number
        @param event_types: event types to which to listen
        @param sources: device ids to which to listen
        @param ctx: a ZMQ context
        
        """
        self._logger = logging.getLogger(__name__ + "." +\
                                         self.__class__.__name__)
        if ctx is None:
            self._ctx = zmq.Context()
        else:
            # For inproc communication.
            self._ctx = ctx
        self._socket = self._ctx.socket(zmq.SUB)
        self._address = protocol+"://"+host
        if port is not None:
            self._address += ":"+str(port)
        self._logger.debug("Address: " + self._address)
        self._socket.connect(self._address)
        self._event_types = event_types
        self._sources = sources
        for event_type in event_types:
            # Various events (motor moved, ...).
            for source in sources:
                # Various devices producing the same event.
                if source == "":
                    # All sources for given event.
                    self._socket.setsockopt(zmq.SUBSCRIBE, str(event_type))
                else:
                    self._socket.setsockopt(zmq.SUBSCRIBE, str(event_type)+
                                        eventformat.separator+str(source))

        # Run a daemon handling the received messages.
        self._runner = Thread(target=self._handle)
        self._runner.daemon = True
        self._runner.start()
        
    def _unpack_event(self, msg):
        event_type, source, event_id, data = msg.split(eventformat.separator)
        event = Event(int(event_type), source, data=data, id=int(event_id))
        self._logger.debug("Received event: %s." % (event))
        
        return event
        
    def _handle(self):
        """Distribute the incoming event to all the handlers."""
        while True:
            event = self._unpack_event(self._socket.recv())
            # Get all classes which this instance implements and call their
            # method to handle the event.
            for cls in self.__class__.__bases__:
                cls._handle_event(self, event)
    
    @abstractmethod
    def _handle_event(self, event):
        """Do something with an incoming event. Every subclass needs to
        implement specific behavior for it depending on the class functionality.
        
        """
        raise NotImplementedError
    
    
class MotionEventListener(_EventListener):
    """Interface specifically designed to serve motion-related events."""
    def __init__(self, protocol, host, port, sources=[""], ctx=None):
        """Constructor.
        
        @param protocol: a ZMQ protocol name
        @param host: host name
        @param port: port number
        @param device_ids: device ids to which to listen
        @param ctx: a ZMQ context
        
        """
        event_types = [eventtype.start,
                       eventtype.stop,
                       eventtype.position_changed,
                       eventtype.state_changed,
                       eventtype.limit_reached]
        super(MotionEventListener, self).__init__(protocol, host, port,
                                                  event_types, sources, ctx)
        
    def _handle_event(self, event):
        if event.event_type == eventtype.start:
            self.on_start(event)
        elif event.event_type == eventtype.stop:
            self.on_stop(event)
        elif event.event_type == eventtype.position_changed:
            self.on_position_changed(event)
        elif event.event_type == eventtype.state_changed:
            self.on_state_changed(event)
        elif event.event_type == eventtype.limit_reached:
            self.on_limit_reached(event)
            
    @abstractmethod
    def on_start(self, event):
        raise NotImplementedError
    
    @abstractmethod
    def on_stop(self, event):
        raise NotImplementedError
    
    @abstractmethod
    def on_state_changed(self, event):
        raise NotImplementedError
    
    @abstractmethod
    def on_position_changed(self, event):
        raise NotImplementedError
    
    @abstractmethod
    def on_limit_reached(self, event):
        raise NotImplementedError