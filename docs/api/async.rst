======================
Asynchronous execution
======================

.. automodule:: concert.asynchronous

Concurrency
===========

Every user defined function or method **must** be synchronous (blocking).
To define a function as asynchronous, use the :func:`.async` decorator::

    from concert.asynchronous import async

    @async
    def synchronous_function():
        # long running operation
        return 1

Every asynchronous function returns a *Future* that can be used for explicit synchronization::

    future = synchronous_function()
    print(future.done())
    result = future.result()

Every future that is returned by Concert, has an additional method ``wait``
that will block until execution finished and raise the exception that might
have been raised in the wrapped function. It will also return the future to
gather the result::

    try:
        future = synchronous_function().wait()
        result = future.result()
    except:
        print("synchronous_function raised an exception")

The asynchronous execution provided by Concert deals with concurrency. If the
user wants to employ real parallelism they should make use of the
multiprocessing module which provides functionality not limited by Python's
global interpreter lock.


.. autofunction:: async
.. autofunction:: is_async
.. autofunction:: wait


Messaging
=========

The backbone of the local messaging system is a dispatching mechanism based on
the publish-subscribe analogy. Once a dispatcher object is created, objects can
:meth:`Dispatcher.subscribe` to messages from other objects and be notified
when other objects :meth:`Dispatcher.send` a message to the dispatcher::

    from concert.asynchronous import Dispatcher

    def handle_message(sender):
        print("{0} send me a message".format(sender))

    dispatcher = Dispatcher()

    obj = {}
    dispatcher.subscribe(obj, 'foo', handle_message)
    dispatcher.send(obj, 'foo')

If not stated otherwise, users should use the global :data:`.dispatcher` for
sending and receiving messages.

.. autoclass:: Dispatcher
    :members:

.. py:data:: concert.asynchronous.dispatcher

    A global :py:class:`Dispatcher` instance used by all devices.
