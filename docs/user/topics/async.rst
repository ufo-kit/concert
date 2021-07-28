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
:py:func:`.start` returns a task_ object, which can also be ``await``\ed.
Overall, in Concert there are three ways to execute coroutines:

    1. as *non-blocking* *tasks*,
    2. via the *blocking* ``await`` syntax,
    3. as a *blocking* *command*.

An example::

    import asyncio
    from concert.coroutines.base import start
    from concert.commands import create_command

    async def corofunc():
        await asyncio.sleep(0.1)
        return 1

    task = start(corofunc()) # doesn't block
    await task # this blocks
    await corofunc() # this blocks too
    cmd = create_command(corofunc) # convenience function
    cmd() # this blocks too


There are many commands which Concert defines for convenience, use the
``cdoc()`` function. A more reallistic example::

    from concert.coroutines.base import start
    from concert.devices.motors.dummy import LinearMotor

    motor = LinearMotor()
    task = start(motor.home()) # this doesn't block
    await task # this blocks
    await motor.home() # this blocks too
    home(motor) # this blocks too, Concert defines the home command

Please note that you cannot use blocking commands from within an ``async def``
function. If you are writing such a function, you must make sure your code is
cooperative and use the ``await`` syntax. The commands are for user convenience
for the command line only. In case you are unsure if a function you are going to
use is a coroutine function, use ``iscoroutinefunction(func)`` test, which
returns ``True`` if you need to use ``func`` with one of the three mechanisms
above. If it returns ``False``, ``func`` is an ordinary function and you can
simply invoke it. For more examples please refer to concert examples_.

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
.. _examples: https://github.com/ufo-kit/concert-examples
.. _gather: https://docs.python.org/3/library/asyncio-task.html#asyncio.gather
