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


def launch(action, args=(), blocking=False):
    """Launch *action* with *args*.

    If *blocking* is ``True``, *action* will be called like an ordinary
    function otherwise a thread will be started. *args* must be a tuple of
    arguments that is then unpacked and passed to *action* at launch time.
    *Action* must be blocking.
    """
    def _action(event, *args):
        action(*args)
        event.set()

    event = Event()
    if blocking:
        _action(event, *args)
    else:
        thread = threading.Thread(target=_action, args=((event,)+args))
        thread.daemon = True
        thread.start()
    return event
