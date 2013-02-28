'''
Created on Feb 28, 2013

@author: farago
'''
import itertools


class Event(object):
    """Class representing an event. It is the basic transfer unit."""
    # A unique id generator.
    event_id = itertools.count().next
    
    def __init__(self, event_type, source, data="", id=None):
        """Constructor.
        
        @param event_type: a unique event type id
        @param source: event source
        @param data: data which the event carries
        
        """
        self._event_type = event_type
        self._source = source
        self._data = data
        if id:
            # The event which we need to reconstruct might already have an id.
            # In that case we use the one already assigned.
            self._id = id
        else:
            self._id = Event.event_id()
        
    @property
    def id(self):
        return self._id
        
    @property
    def event_type(self):
        return self._event_type
    
    @property
    def source(self):
        return self._source
    
    @property
    def data(self):
        return self._data
    
    def __str__(self):
        return "Event(id=%d, type=%d, source=%s, data=%s)" %\
            (self.id, self.event_type, self.source, self.data == "")