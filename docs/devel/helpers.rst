=======
Helpers
=======

Messaging
=========

The backbone of the local messaging system is a dispatching mechanism based on
the publish-subscribe analogy. Once a dispatcher object is created, objects can
:meth:`Dispatcher.subscribe` to messages from other objects and be notified
when other objects :meth:`Dispatcher.send` a message to the dispatcher::

    from concert.async import Dispatcher

    def handle_message(sender):
        print("{0} send me a message".format(sender))

    dispatcher = Dispatcher()

    obj = {}
    dispatcher.subscribe(obj, 'foo', handle_message)
    dispatcher.send(obj, 'foo')

If not stated otherwise, users should use the global :data:`.dispatcher` for
sending and receiving messages.

.. py:data:: concert.helpers.dispatcher

    A global :py:class:`Dispatcher` instance used by all devices.
