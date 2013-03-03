'''
Created on Mar 3, 2013

@author: farago
'''
from controlledobject import Identifiable

class Event(Identifiable):
    def __init__(self, event_type, source, data=None):
        super(Event, self).__init__()
        self._type = event_type
        self._source = source
        self._data = data
        
    @property
    def event_type(self):
        return self._type
    
    @property
    def source(self):
        return self._source
    
    @property
    def data(self):
        return self._data
        
    def __str__(self):
        return "Event(type=%s, source=%s, data=%s)" % (self._type,
                                                       self._source,
                                                       str(self._data))

    def __repr__(self):
        return str(self)