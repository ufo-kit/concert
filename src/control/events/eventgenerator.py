'''
Created on Mar 3, 2013

@author: farago
'''
import Queue
import logging
from threading import Thread

logger = logging.getLogger(__name__)
_event_generator = None

class EventGenerator(object):
    def __init__(self):
        self._logger = logging.getLogger(__name__ + "." +\
                                         self.__class__.__name__)
        self._listeners = []
        self._events = Queue.Queue()
        
        self._runner = Thread(target=self._serve)
        self._runner.daemon = True
        self._runner.start()
    
    def add_listener(self, listener):
        self._listeners.append(listener)
        
    def _enqueue(self, event):
        self._logger.debug("Queueing event: " + repr(event))
        self._events.put(event)
        
    def _serve(self):
        while True:
            event = self._events.get()
            for listener in self._listeners:
                listener._handle(event)
            self._events.task_done()

def get_generator():
    global _event_generator
    if _event_generator == None:
        _event_generator = EventGenerator()
    
    return _event_generator

def fire(event):
    get_generator()._enqueue(event)