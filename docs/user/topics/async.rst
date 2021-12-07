.. _concurrent-execution:


====================
Concurrent execution
====================

Concert relies on concurrency_ instead of parallelism_ because what mostly
happens is communication with devices, which is I/O bound. Concurrency is
realized via *coroutines* and Python's asyncio_ module. A *coroutine* is a
function defined as ``async def`` and inside it can yield execution for other
coroutines via the ``await`` keyword. When you call a *coroutine function*, it
returns a *coroutine* object, but the code of that function is not yet executed.
One way of invoking execution in a *blocking* way is by the ``await`` keyword
followed by a coroutine object. This will block the session until the coroutine
is finished.  Alternatively, you can start the execution in a *non-blocking* way
by calling :py:func:`.start` and get the control back immediately.
:py:func:`.start` returns a task_ object, which can also be ``await``\ed. Most
of the ``async def`` functions in Concert are wrapped into tasks by the
:py:func:`.background` decorator, so you do not need to use the
:py:func:`.start` in order to start execution immediately. You should however
keep this in mind when writing your own coroutines and decorate them (see below)
if you want them to be automatically started upon invocation.  Overall, in
Concert there are two ways to execute coroutines:

    1. as *non-blocking* *tasks*,
    2. via the *blocking* ``await`` syntax,

An example::

    import asyncio
    from concert.coroutines.base import background, start

    async def corofunc():
        await asyncio.sleep(0.1)
        return 1

    @background
    async def corofunc_run_immediately():
        await asyncio.sleep(0.1)
        return 1

    coro = corofunc() # coro is a coroutine, not yet a task and has not started
    task = start(coro) # wraps the coroutine into a task and starts it, does not block
    result = await task # this blocks, result contains 1
    await corofunc() # this blocks too

    task = corofunc_run_immediately() # runs immediately, does not block
    result = await task # this blocks, result contains 1
    await corofunc_run_immediately() # this blocks too


A more reallistic example::

    from concert.devices.motors.dummy import LinearMotor

    motor = LinearMotor()
    task = motor.home() # this doesn't block
    await task # this blocks
    await motor.home() # this blocks too

You can cancel running coroutines which are being ``await``\ed by pressing
*ctrl-c*. This for instance stops a motor. If you want to cancel *all* running
coroutines, including the ones running in the background, press *ctrl-k*.


Concurrency
-----------

Concurrent execution itself is realized via asyncio's tools, like gather_, which
executes given coroutines concurrently and returns their results::

    async def corofunc():
        await asyncio.sleep(0.1)
        return 1

    await asyncio.gather(corofunc(), corofunc())


Synchronization
---------------

When using the concurrent getters and setters of :class:`.Device` and
:class:`.Parameter`, coroutines can not be sure if other coroutines manipulate
the device. To lock devices or specific parameters, coroutines can use devices
with context managers::

    async with shutter, motor['position']:
        await motor.set_position(2 * q.mm)
        await shutter.open()

Inside the ``async with`` environment, a coroutine has exclusive access to the devices
and parameters.

.. _concurrency: https://en.wikipedia.org/wiki/Concurrency_(computer_science)
.. _parallelism: https://en.wikipedia.org/wiki/Parallel_computing
.. _asyncio: https://docs.python.org/3/library/asyncio.html
.. _task: https://docs.python.org/3/library/asyncio-task.html#task-object
.. _gather: https://docs.python.org/3/library/asyncio-task.html#asyncio.gather
