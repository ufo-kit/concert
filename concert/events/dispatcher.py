import threading
import Queue


class Dispatcher(object):
    """Core dispatcher"""

    def __init__(self):
        self._subscribers = {}
        self._messages = Queue.Queue()
        self._event_queues = {}
        self._lock = threading.Lock()

        server = threading.Thread(target=self._serve)
        server.daemon = True
        server.start()

    def subscribe(self, events, handler):
        """Subscribe to a message sent by sender.

        When message is sent by sender, handler is called with sender as the
        only argument.

        """
        for t in events:
            if t in self._subscribers:
                self._subscribers[t].add(handler)
            else:
                self._subscribers[t] = set([handler])

    def unsubscribe(self, events, handler):
        for t in events:
            if t in self._subscribers:
                self._subscribers[t].remove(handler)

    def send(self, sender, message):
        """Send message from sender."""
        self._messages.put((sender, message))

    def wait(self, events, timeout=None):
        """Wait until sender sent message."""
        for event in events:
            event.wait(timeout)

    def _serve(self):
        while True:
            t = self._messages.get()
            sender, message = t

            if t in self._subscribers:
                for callback in self._subscribers[t]:
                    callback(sender)
            if (None, message) in self._subscribers:
                for callback in self._subscribers[(None, message)]:
                    callback(sender)

            if t in self._event_queues:
                self._event_queues[t].notify_and_clear()

            self._messages.task_done()


dispatcher = Dispatcher()
