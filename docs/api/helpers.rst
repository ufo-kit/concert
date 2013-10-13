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


Coroutines
==========

Coroutines provide a way to process data, yield execution until more data
is produced and so on. They are used together with generators to provide
such behavior. If one wants to program a coroutine then it must be
decorated with :py:func:`coroutine`.

.. autofunction:: coroutine

To connect a generator and a coroutine Concert provides an :py:func:`inject`
function which lets the generator produce the data and feeds a consumer with
it.

.. autofunction:: inject

An example generator and coroutine pair could look like this::

    from concert.helpers import coroutine, inject


    def generate():
        for i in range(3):
            yield i


    @coroutine
    def consume():
        while True:
            item = yield
            print(item)

The generator produces some numbers and the consumer just prints them to
stdout. To connect these two we use the :py:func:`inject` function,
so the execution of the code could look like::

    inject(generate(), consume())

This will print::

    0
    1
    2

If there are more than one consumer which want to get the data, one can use the
:py:func:`broadcast` which maps 1 source to N consumers.

.. autofunction:: broadcast
 

The generators and coroutines yield execution, but if the data production
should not be stalled by data consumption the coroutine should only provide
data buffering and delegate the real consumption to a separate thread or
process. The same can be achieved by first buffering the data and then
yielding them by a generator. It comes from the fact that a generator
will not produce a new value until the old one has been consumed.


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

.. autoclass:: Dispatcher
    :members:

.. py:data:: concert.helpers.dispatcher

    A global :py:class:`Dispatcher` instance used by all devices.

