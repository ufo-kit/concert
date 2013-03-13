======================
The message dispatcher
======================

The backbone of the local event system is a dispatching mechanism based on
the publish-subscribe analogy. Once a dispatcher object is created, objects can
subscribe to messages from other objects and are notified when the message is
sent to the dispatcher::

    from concert.events.dispatcher import Dispatcher

    def handle_message(sender):
        print("{0} send me a message".format(sender))

    dispatcher = Dispatcher()

    obj = {}
    dispatcher.subscribe([(obj, 'foo')], handle_message)
    dispatcher.send(obj, 'foo')


Programming interface
=====================

.. automodule:: concert.events.dispatcher
    :members:

.. py:data:: concert.events.dispatcher.dispatcher

    A global :py:class:`Dispatcher` instance used by all devices.

