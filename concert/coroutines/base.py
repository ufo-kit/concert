import asyncio
import concert.config
import concurrent.futures
import functools
import queue
import logging
import time
from threading import Thread
from concert.config import AIODEBUG
from concert.helpers import PrioItem
from concert.quantities import q


LOG = logging.getLogger(__name__)
# Loop which runs in a separate thread
_TLOOP = None


async def async_generate(iterable):
    for item in iterable:
        yield item


def run_in_loop(coroutine):
    """Wrap *coroutine* into a `asyncio.Task`, run it in the current loop, block until it finishes
    and return the result.  On KeyboardInterrupt, the task is cancelled.
    """
    loop = asyncio.get_event_loop()
    task = asyncio.ensure_future(coroutine, loop=loop)

    try:
        result = loop.run_until_complete(task)
        return result
    except KeyboardInterrupt:
        LOG.log(AIODEBUG, "KeyboardInterrupt in `%s', cancelling", task.get_coro().__qualname__)
        task.cancel()


def run_in_loop_thread_blocking(coroutine):
    """Run *coroutine* in a new loop in a separate thread. Useful for third party functions, which
    are unaware of asyncio and we want to feed them with our asyncio code (like in the optimization
    module).
    """
    global _TLOOP

    if _TLOOP:
        raise RuntimeError('Another coroutine is already running in a separate thread. '
                           'Currently only one coroutine can run in a separate thread at a time.')

    def run():
        _TLOOP.run_forever()

    _TLOOP = asyncio.new_event_loop()
    thread = Thread(target=run, daemon=False)
    thread.start()

    try:
        future = asyncio.run_coroutine_threadsafe(coroutine, _TLOOP)
        return future.result()
    except concurrent.futures.CancelledError as e:
        LOG.log(AIODEBUG, 'run_in_loop_thread_blocking: %s cancelled', coroutine.__qualname__)
    finally:
        _TLOOP.call_soon_threadsafe(_TLOOP.stop)
        thread.join()
        _TLOOP.close()
        _TLOOP = None


def run_in_executor(func, *args):
    """Run a blocking function *func* with signature func(\*args) in an executor."""
    # Leave this here for now, if there are problems with the normal run_in_executor returning a
    # future then let's re-visit
    # """Run a blocking function *func* with signature func(*args) in an executor, wrap the future
    # into a real coroutine and return it.
    # """
    # async def make_coro():
    #     loop = asyncio.get_running_loop()
    #     future = loop.run_in_executor(None, func, *args)
    #
    #     try:
    #         return await future
    #     except asyncio.CancelledError:
    #         # Wait for the future to finish even if we are cancelled.
    #         LOG.log(AIODEBUG, f'{func.__name__} waiting for concurrent.Future')
    #         return await future
    # return make_coro()
    loop = asyncio.get_event_loop()

    return loop.run_in_executor(None, func, *args)


def start(coroutine):
    """Wrap *coroutine* into a task and start its execution right away. The returned task will also
    be cancellable by ctrl-k.
    """
    task = asyncio.ensure_future(coroutine)
    task._is_concert_task = True

    return task


def background(coroutine):
    """Same as :func:`.start`, just meant to be used as a decorator."""
    @functools.wraps(coroutine)
    def inner(*args, **kwargs):
        return start(coroutine(*args, **kwargs))

    return inner


async def ensure_coroutine(func, *args, **kwargs):
    """func(\*args, \*\*kwargs) returns an awaitable which is wrapped here into a real coroutine.
    This is useful for turuning futures from other libraries, like Tango, into real coroutines.
    """
    return await func(*args, **kwargs)


def broadcast(producer, *consumers):
    """
    broadcast(producer, *consumers)

    Feed *producer* to all *consumers*.
    """
    loop = asyncio.get_event_loop()
    next_val = loop.create_future()
    consumed = loop.create_future()
    stop = object()
    num_consumers = len(consumers)
    used = num_consumers

    async def produce():
        try:
            async for elem in producer:
                if not next_val.cancelled():
                    next_val.set_result(elem)
                await consumed
                if not num_consumers:
                    break
            if not next_val.cancelled():
                next_val.set_result(stop)
        except Exception as e:
            LOG.warn(f"Exception `{e}' in broadcast")
            # Make sure the duplicates stop even if there is an exception
            if not next_val.cancelled():
                next_val.set_result(stop)
            raise

    async def duplicate():
        nonlocal next_val, consumed, used, num_consumers
        try:
            while True:
                val = await next_val
                if val is stop:
                    return
                yield val
                used -= 1
                if not used:
                    if not consumed.cancelled():
                        consumed.set_result(None)
                        consumed = loop.create_future()
                        next_val = loop.create_future()
                    used = num_consumers
                else:
                    await consumed
        except Exception as e:
            print('broadcast duplicate cancelled', e)
            raise
        finally:
            # Our consumer called break in its async for, remove it from waiting list
            num_consumers -= 1
            used -= 1
            if not used:
                if not consumed.cancelled():
                    consumed.set_result(None)
                    consumed = loop.create_future()
                    next_val = loop.create_future()
                used = num_consumers

    started = [produce()]
    for consumer in consumers:
        started.append(consumer(duplicate()))

    return started


async def feed_queue(producer, func, *args):
    """Feed function *func* with items from *producer* in a separete thread. The signatute must be
    func(queue, \*args) where elements in the queue are instances of
    :class:`concert.helpers.PrioItem`.
    """
    loop = asyncio.get_running_loop()
    pqueue = queue.PriorityQueue()

    try:
        future = loop.run_in_executor(None, func, pqueue, *args)
        prio = 1
        async for item in producer:
            pqueue.put(PrioItem(priority=prio, data=item))
            prio += 1
    except (asyncio.CancelledError, KeyboardInterrupt) as e:
        LOG.log(concert.config.AIODEBUG, f'feed_queue cancelled by exception {type(e)}')
        # Highest priority, effectively cancel processing of everything in the queue
        prio = 0
        raise
    finally:
        LOG.log(concert.config.AIODEBUG, f'feed_queue finished with priority {prio}')
        pqueue.put(PrioItem(priority=prio, data=None))
        await future


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
