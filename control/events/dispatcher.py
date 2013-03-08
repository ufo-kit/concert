import threading
import Queue


class LockedEventList(object):
    def __init__(self, events=[]):
        self._lock = threading.Lock()
        self._events = events

    def add_event(self, event):
        self._lock.acquire()
        self._events.append(event)
        self._lock.release()

    def notify_and_clear(self):
        self._lock.acquire()
        for event in self._events:
            event.set()

        self._events = []
        self._lock.release()


class Dispatcher(object):
    def __init__(self):
        self._subscribers = {}
        self._messages = Queue.Queue()
        self._events = {}

        server = threading.Thread(target=self._serve)
        server.daemon = True
        server.start()

    def subscribe(self, sender, message, handler):
        """Subscribe to a message sent by sender.

        When message is sent by sender, handler is called with sender as the
        only argument.

        """
        t = (sender, message)
        try:
            self._subscribers[t].add(handler)
        except KeyError:
            self._subscribers[t] = set([handler])

    def send(self, sender, message):
        """Send message from sender."""
        self._messages.put((sender, message))

    def wait(self, sender, message, timeout=None):
        """Wait until sender sent message."""
        t = (sender, message)
        event = threading.Event()

        if t in self._events:
            self._events[t].add_event(event)
        else:
            self._events[t] = LockedEventList([event])

        event.wait(timeout)

    def _serve(self):
        while True:
            t = self._messages.get()
            sender, message = t

            if t in self._subscribers:
                for callback in self._subscribers[t]:
                    callback(sender)

            if t in self._events:
                self._events[t].notify_and_clear()

            self._messages.task_done()


dispatcher = Dispatcher()
