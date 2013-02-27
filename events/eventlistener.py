'''
Created on Feb 27, 2013

@author: farago
'''
import zmq
import eventformat
import eventtype
from threading import Thread
import logging


class _EventListener(object):
    """Interface for listening to events. Must be inherited from. Uses ZMQ
    for transporting messages. A listener can listen to various events from
    various sources. The protocol follows publisher-subscriber pattern. This
    interface is private to this module and should not be implemented from
    outside. There are follow-up specific interfaces which are supposed to
    be implemented outside this module.
    """
    def __init__(self, protocol, host, port, event_types,
                                                device_ids=[""],  ctx=None):
        """Constructor.
        
        @param protocol: a ZMQ protocol name
        @param host: host name
        @param port: port number
        @param event_types: event types to which to listen
        @param device_ids: device ids to which to listen
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
        self._source_ids = device_ids
        for event_type in event_types:
            # Various events (motor moved, ...).
            for source_id in device_ids:
                # Various devices producing the same event.
                if source_id == "":
                    # All sources for given event.
                    self._socket.setsockopt(zmq.SUBSCRIBE, str(event_type))
                else:
                    self._socket.setsockopt(zmq.SUBSCRIBE, str(event_type)+
                                        eventformat.separator+str(source_id))

        # Run a daemon handling the received messages.
        self._runner = Thread(target=self._handle)
        self._runner.daemon = True
        self._runner.start()
        
    def _unpack_message(self, msg):
        event_type, source_id, event_id, data = msg.split(eventformat.separator)
        self._logger.debug("Received message (first KB): %s" % (msg[:1024]))
        
        return int(event_type), source_id, int(event_id), data
        
    def _handle(self):
        raise NotImplementedError
    
    
class MotionEventListener(_EventListener):
    """Interface specifically designed to serve motion-related events."""
    def __init__(self, protocol, host, port, device_ids=[""], ctx=None):
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
                                                  event_types, device_ids, ctx)
        
    def _handle(self):
        while True:
            event_type, source_id, event_id, data =\
                            self._unpack_message(self._socket.recv())
            if event_type == eventtype.start:
                self.on_start(source_id, event_id, data)
            elif event_type == eventtype.stop:
                self.on_stop(source_id, event_id, data)
            elif event_type == eventtype.position_changed:
                self.on_position_changed(source_id, event_id, data)
            elif event_type == eventtype.state_changed:
                self.on_state_changed(source_id, event_id, data)
            elif event_type == eventtype.limit_reached:
                self.on_limit_reached(source_id, event_id, data)
            
    def on_start(self, source_id, event_id, data):
        raise NotImplementedError
    
    def on_stop(self, source_id, event_id, data):
        raise NotImplementedError
    
    def on_state_changed(self, source_id, event_id, data):
        raise NotImplementedError
    
    def on_position_changed(self, source_id, event_id, data):
        raise NotImplementedError
    
    def on_limit_reached(self, source_id, event_id, data):
        raise NotImplementedError