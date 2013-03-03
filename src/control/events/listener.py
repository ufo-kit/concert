'''
Created on Mar 3, 2013

@author: farago
'''
from abc import ABCMeta, abstractmethod
import eventgenerator
import eventtype

class EventListener(object):
    __metaclass__ = ABCMeta
    
    def __init__(self):
        eventgenerator.get_generator().add_listener(self)
    
    @property    
    def event_types(self):
        events = []
        for cls in self.__class__.__bases__:
            events += cls._EVENT_TYPES
            
        return events
        
    def _handle(self, event):
        """Distribute the incoming event to all the handlers."""
        # Get all classes which this instance implements and call their
        # method to handle the event.
        for cls in self.__class__.__bases__:
            if event.event_type in self.event_types:
                cls._handle_event(self, event)
                
    @abstractmethod
    def _handle_event(self, event):
        """Do something with an incoming event. Every subclass needs to
        implement specific behavior for it depending on the class functionality.
        
        """
        raise NotImplementedError
    
class MotionEventListener(EventListener):
    _EVENT_TYPES = [eventtype.Motion.START,
                    eventtype.Motion.STOP,
                    eventtype.Motion.POSITION_CHANGED,
                    eventtype.Motion.LIMIT_REACHED]
    
    def __init__(self):
        super(MotionEventListener, self).__init__()
        
    def _handle_event(self, event):
        if event.event_type == eventtype.Motion.START:
            self.on_start(event)
        elif event.event_type == eventtype.Motion.STOP:
            self.on_stop(event)
        elif event.event_type == eventtype.Motion.POSITION_CHANGED:
            self.on_position_changed(event)
        elif event.event_type == eventtype.Motion.LIMIT_REACHED:
            self.on_limit_reached(event)
            
    def on_start(self, event):
        pass
    
    def on_stop(self, event):
        pass

    def on_position_changed(self, event):
        pass
    
    def on_limit_reached(self, event):
        pass