'''
Created on Mar 3, 2013

@author: farago
'''
from control.events import generator as eventgenerator
from control.devices.motion.axes.axis import AxisState
from control.events import type as eventtype


class EventListener(object):
    def __init__(self):
        eventgenerator.get_generator().add_listener(self)
    
    @property    
    def event_types(self):
        events = []
        for cls in self.__class__.__bases__:
            events += cls._EVENT_TYPES
            
        return events + self.__class__._EVENT_TYPES
        
    def _handle(self, event):
        """Distribute the incoming event to all the handlers."""
        # Get all classes which this instance implements and call their
        # method to handle the event.
        for cls in self.__class__.__bases__:
            if event.event_type in self.event_types:
                cls._handle_event(self, event)
                
    def _handle_event(self, event):
        """Do something with an incoming event. Every subclass needs to
        implement specific behavior for it depending on the class functionality.
        
        """
        raise NotImplementedError


class StateChangeListener(EventListener):
    _EVENT_TYPES = [eventtype.StateChangeEvent.STATE]
    
    def __init__(self):
        super(StateChangeListener, self).__init__()
    
    def _handle_event(self, event):
        if event.event_type == eventtype.StateChangeEvent.STATE:
            self._handle_state_change(event.source, event.data)
            
    def _handle_state_change(self, source, state):
        """Handle a particular state change."""
        raise NotImplementedError
    

class AxisStateListener(StateChangeListener):
    def __init__(self):
        super(AxisStateListener, self).__init__()
    
    def _handle_state_change(self, source, state):
        if state == AxisState.MOVING:
            self.on_moving(source)
        elif state == AxisState.POSITION_LIMIT:
            self.on_position_limit(source)
        elif state == AxisState.STANDBY:
            self.on_standby(source)
            
    def on_moving(self, source):
        raise NotImplementedError
    
    def on_position_limit(self, source):
        raise NotImplementedError
    
    def on_standby(self, source):
        raise NotImplementedError