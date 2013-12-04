======================
Asynchronous execution
======================

.. automodule:: concert.helpers

Concurrency
===========

Every user defined function or method **must** be synchronous (blocking).
To define a function as asynchronous, use the :func:`.async` decorator::

    from concert.helpers import async

    @async
    def synchronous_function():
        # long running operation
        return 1

Every asynchronous function returns a *Future* that can be used for explicit
synchronization::

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


Synchronization
---------------

When using the asynchronous getters and setters of :class:`Device` and
:class:`Parameter`, processes can not be sure if other processes or the user
manipulate the device during the execution. To lock devices or specific
parameters, processes can use them as context managers::

    with motor, pump['foo']:
        motor.position = 2 * q.mm
        pump.foo = 1 * q.s

Inside the ``with`` environment, the process has exclusive access to the devices
and parameters.


Testing
-------

Testing and debugging asynchronous code can be difficult at times because the
real source of an error is hidden behind calls from different places. To
disable asynchronous execution (but still keeping the illusion of having
Futures returned), you can import :data:`.DISABLE` and set it to ``True``
*before* importing anything else from Concert.

Concert already provides a Nose plugin that adds a ``--disable-async`` flag to
the test runner which in turn sets :data:`.DISABLE` to ``True``.

.. py:data:: concert.helpers.DISABLE

    A global configuration variable that will disable asynchronous execution
    when set to ``True``.


Messaging
=========

The backbone of the local messaging system is a dispatching mechanism based on
the publish-subscribe analogy. Once a dispatcher object is created, objects can
:meth:`Dispatcher.subscribe` to messages from other objects and be notified
when other objects :meth:`Dispatcher.send` a message to the dispatcher::

    from concert.helpers import Dispatcher

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
