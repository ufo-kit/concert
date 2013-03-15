"""
The mother of all bases. The lowest level object definition and functionality.
"""
import threading
from concert.events.dispatcher import dispatcher
from threading import Event


class ConcertObject(object):
    """
    Base concert object which can send, subscribe to and wait for messages.
    """
    def send(self, message):
        """Send a *message* tied with this object."""
        dispatcher.send(self, message)

    def subscribe(self, message, callback):
        """Subscribe to a *message* from this object.

        *callback* will be called with this object as the first argument.
        """
        dispatcher.subscribe([(self, message)], callback)

    def unsubscribe(self, message, callback):
        """
        Unsubscribe from a *message*.

        *callback* is a function which is unsubscribed from a particular
        *message* coming from this object.
        """
        dispatcher.unsubscribe([(self, message)], callback)

    def _launch(self, param, action, args=(), blocking=False):
        """Launch *action* with *args* with message and event handling after
        the action finishes.

        If *blocking* is ``True``, *action* will be called like an ordinary
        function otherwise a thread will be started. *args* must be a tuple of
        arguments that is then unpacked and passed to *action* at _launch time.
        The *action* itself must be blocking.
        """
        def _action(event, args):
            """Call action and handle its finish."""
            action(*args)
            event.set()
            self.send(getattr(self.__class__, param.upper()))

        event = Event()
        launch(_action, (event, args), blocking)
        return event


def launch(action, args=(), blocking=False):
    """Launch *action* with *args*.

    If *blocking* is ``True``, *action* will be called like an ordinary
    function otherwise a thread will be started. *args* must be a tuple of
    arguments that is then unpacked and passed to *action* at launch time.
    The *action* itself must be blocking.
    """
    if blocking:
        action(*args)
    else:
        thread = threading.Thread(target=action, args=args)
        thread.daemon = True
        thread.start()
