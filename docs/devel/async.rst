======================
Asynchronous execution
======================

Concurrency
===========

Every user defined function or method *must* be synchronous (blocking). To
define a function as asynchronous, use the :func:`.async` decorator::

    from concert.async import async

    @async
    def synchronous_function():
        # long running operation
        return 1

Every asynchronous function returns a *Future* that can be used for explicit
synchronization::

    future = synchronous_function()
    print(future.done())
    result = future.result()

Every future that is returned by Concert, has an additional method ``join``
that will block until execution finished and raise the exception that might
have been raised in the wrapped function. It will also return the future to
gather the result::

    try:
        future = synchronous_function().join()
        result = future.result()
    except:
        print("synchronous_function raised an exception")

The asynchronous execution provided by Concert deals with concurrency. If the
user wants to employ real parallelism they should make use of the
multiprocessing module which provides functionality not limited by Python's
global interpreter lock.


Synchronization
---------------

When using the asynchronous getters and setters of :class:`.Device` and
:class:`.Parameter`, processes can not be sure if other processes or the user
manipulate the device during the execution. To lock devices or specific
parameters, processes can use them as context managers::

    with motor, pump['foo']:
        motor.position = 2 * q.mm
        pump.foo = 1 * q.s

Inside the ``with`` environment, the process has exclusive access to the devices
and parameters.


Disable asynchronous execution
------------------------------

Testing and debugging asynchronous code can be difficult at times because the
real source of an error is hidden behind calls from different places. To disable
asynchronous execution (but still keeping the illusion of having Futures
returned), you can import :data:`.DISABLE_ASYNC` and set it to ``True`` *before*
importing anything else from Concert.

Concert already provides a Nose plugin that adds a ``--disable-async`` flag to
the test runner which in turn sets :data:`.DISABLE_ASYNC` to ``True``.
