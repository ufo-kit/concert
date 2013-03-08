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
    """Generates events. It is a singleton which distributes events among
    listeners who want to listen to them.

    """
    def __init__(self):
        """Constructor."""
        self._logger = logging.getLogger(__name__ + "." +\
                                         self.__class__.__name__)
        self._listeners = set([])
        self._events = Queue.Queue()

        self._runner = Thread(target=self._serve)
        self._runner.daemon = True
        self._runner.start()

    def add_listener(self, listener):
        """Add listener.

        @param listener: listener

        """
        self._listeners.add(listener)

    def remove_listener(self, listener):
        """Remove listener.

        @param listener: listener

        """
        if listener in self._listeners:
            self._listeners.remove(listener)

    def _enqueue(self, event):
        """Enqueue an event into a processing queue for processing.

        @param event: event

        """
        self._logger.debug("Queueing event: " + repr(event))
        self._events.put(event)

    def _serve(self):
        """Serve forever. Distribute events among listeners. This method is run
        in a separate thread.

        """
        while True:
            event = self._events.get()
            for listener in self._listeners:
                listener._handle(event)
            self._events.task_done()


def get_generator():
    """Access to the event generator singleton.

    @return: event generator

    """
    global _event_generator
    if _event_generator == None:
        _event_generator = EventGenerator()

    return _event_generator


def fire(event):
    """Fire an event.

    @param event: event

    """
    get_generator()._enqueue(event)
