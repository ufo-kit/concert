import asyncio
import concert.config
import functools
import queue
import logging
import time
from concert.config import AIODEBUG
from concert.helpers import PrioItem
from concert.quantities import q


LOG = logging.getLogger(__name__)


def run_in_loop(coroutine):
    """Wrap *coroutine* into a `asyncio.Task`, run it in the current loop and return the result.
    On KeyboardInterrupt, the task is cancelled.
    """
    loop = asyncio.get_event_loop()
    task = loop.create_task(coroutine)

    try:
        result = loop.run_until_complete(task)
        return result
    except KeyboardInterrupt:
        LOG.log(AIODEBUG, "KeyboardInterrupt in `%s', cancelling", task.get_coro().__qualname__)
        task.cancel()


def start(coroutine):
    """Start *coroutine*'s execution right away. This is supposed to be used for user convenience,
    e.g. when they want to start some action without having to deal with asyncio, e.g.
    motor.set_position(pos).

    This calls `asyncio.ensure_future` on the *coroutine* which wraps it into a `asyncio.Task`. We
    have to use this function instead of `asyncio.creat_task`, otherwise IPython would not start its
    execution immediately (as of IPython version 7.22)
    """
    return asyncio.ensure_future(coroutine)


# TODO: test
def inloop(coroutine_func):
    @functools.wraps(coroutine_func)
    def inner(*args, **kwargs):
        LOG.log(concert.config.AIODEBUG, 'inloop: %s', coroutine_func)
        return run_in_loop(coroutine_func(*args, **kwargs))

    return inner


# TODO: test
def broadcast(producer, *consumers):
    """
    broadcast(producer, *consumers)

    Feed *producer* to all *consumers*.
    """
    loop = asyncio.get_event_loop()
    next_val = loop.create_future()
    consumed = loop.create_future()
    stop = object()
    used = len(consumers)  # number of consumers

    async def produce():
        async for elem in producer:
            next_val.set_result(elem)
            await consumed
        next_val.set_result(stop)

    async def duplicate():
        nonlocal next_val, consumed, used
        while True:
            val = await next_val
            if val is stop:
                return
            yield val
            used -= 1
            if not used:
                consumed.set_result(None)
                consumed = loop.create_future()
                next_val = loop.create_future()
                used = len(consumers)
            else:
                await consumed

    started = [produce()]
    for consumer in consumers:
        started.append(consumer(duplicate()))

    return started


# TODO: test
async def feed_queue(producer, func, *args):
    """Feed function *func* with items from *producer* in a separete thread. The signatute must be
    func(queue, *args) where elements in the queue are instances of `concert.helpers.PrioItem`.
    """
    loop = asyncio.get_running_loop()
    pqueue = queue.PriorityQueue()
    future = loop.run_in_executor(None, func, pqueue, *args)

    try:
        prio = 1
        async for item in producer:
            pqueue.put(PrioItem(priority=prio, data=item))
            prio += 1
    except (asyncio.CancelledError, KeyboardInterrupt) as e:
        LOG.log(concert.config.AIODEBUG, f'feed_queue cancelled by exception {type(e)}')
        # Highest priority, effectively cancel processing of everything in the queue
        prio = 0
    finally:
        LOG.log(concert.config.AIODEBUG, f'feed_queue finished with priority {prio}')
        pqueue.put(PrioItem(priority=prio, data=None))
        await future


# TODO: test
async def wait_until(condition, sleep_time=1e-1 * q.s, timeout=None):
    """Wait until a callable *condition* returns True. *sleep_time* is the time to sleep
    between consecutive checks of *condition*. If *timeout* is given and the *condition* doesn't
    return True within the time specified by it a :class:`.WaitingError` is raised.
    """
    sleep_time = sleep_time.to(q.s).magnitude
    if timeout:
        start = time.time()
        timeout = timeout.to(q.s).magnitude

    while not await condition():
        if timeout and time.time() - start > timeout:
            raise WaitError('Waiting timed out')
        await asyncio.sleep(sleep_time)


class WaitError(Exception):

    """Raised on busy waiting timeouts"""

    pass
