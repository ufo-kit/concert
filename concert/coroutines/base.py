import atexit
import asyncio
import concert.config
import functools
import queue
import logging
import threading
from concert.helpers import PrioItem


LOG = logging.getLogger(__name__)


def _shutdown_loop(loop, thread):
    LOG.log(concert.config.AIODEBUG, 'Closing loop')
    # Stop works only via threadsafe
    loop.call_soon_threadsafe(loop.stop)
    # Make sure we don't attempt to close a running loop
    while loop.is_running():
        pass
    # Close works only via direct call
    loop.close()
    LOG.log(concert.config.AIODEBUG, 'Loop closed')
    thread.join()
    LOG.log(concert.config.AIODEBUG, 'Loop thread joined')


def _serve(loop):
    LOG.log(concert.config.AIODEBUG, 'Start loop')
    asyncio.set_event_loop(loop)
    loop.run_forever()


_LOOP = asyncio.new_event_loop()
_THREAD = threading.Thread(target=_serve, args=(_LOOP,), daemon=True)
_THREAD.start()
atexit.register(_shutdown_loop, _LOOP, _THREAD)


def inloop(coroutine_func):
    @functools.wraps(coroutine_func)
    def inner(*args, **kwargs):
        return asyncio.run_coroutine_threadsafe(coroutine_func(*args, **kwargs), _LOOP)

    return inner


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
