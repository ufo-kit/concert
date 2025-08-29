"""
NEVER USE THIS MODULE DIRECTLY!

The only use case of this module is with IPython's "exec_files" managed in bin/concert. The module
contains IPython configuration which must be executed as code and cannot be passsed as config.
"""
import asyncio
import logging
import traceback
from concert.base import (UnitError, LimitError, ParameterError,
                          ReadAccessError, WriteAccessError, LockError)
from concert.coroutines.base import get_event_loop
from concert.session.utils import abort_awaiting, run_exit_functions


LOG = logging.getLogger(__name__)


def _handler(_shell, _etype, evalue, _traceback_, tb_offset=None):
    if _etype == asyncio.CancelledError:
        # Silently log the error message without polluting the terminal
        messages = traceback.extract_tb(_traceback_).format()
        if len(messages) > 1:
            messages = messages[1:]
        LOG.info('asyncio.CancelledError\n' + ''.join(messages))
        print('Cancelled')
    elif _etype == KeyboardInterrupt:
        # Silently log the error message without polluting the terminal
        messages = traceback.extract_tb(_traceback_).format()
        if len(messages) > 2:
            # Skip IPython part and our sigint handler
            messages = messages[1:-1]
        LOG.info('KeyboardInterrupt in normal mode\n' + ''.join(messages))
        print('KeyboardInterrupt')
    else:
        print("Sorry, {0}".format(str(evalue)))
    return None


def _abort_all_awaiting(event):
    abort_awaiting()


async def _ctrl_d(event):
    await run_exit_functions()
    ip = get_ipython()
    ip.ask_exit()
    event.app.exit()


def _pre_run_cell(info):
    from concert.coroutines.base import run_in_loop
    if info.raw_cell in ["exit", "quit"]:
        run_in_loop(run_exit_functions())


_custom_exceptions = (
    UnitError,
    LimitError,
    ParameterError,
    ReadAccessError,
    WriteAccessError,
    LockError,
    KeyboardInterrupt,
    asyncio.CancelledError
)


class ConcertAsyncioRunner:
    def __call__(self, coro):
        loop = get_event_loop()
        task = loop.create_task(coro)
        try:
            return loop.run_until_complete(task)
        except KeyboardInterrupt:
            # Transform KeyboardInterrupt into asyncio.CancelledError
            LOG.debug('KeyboardInterrupt in ConcertAsyncioRunner')
            task.cancel()
            while not task.done():
                # If the task runs some blocking code in an executor we need to wait for it to
                # finish.  Moreover, prevent interruption by multiple ctrl-c presses by the while
                # loop.
                try:
                    loop.run_until_complete(asyncio.wait([task]))
                except KeyboardInterrupt:
                    print('Waiting for task to finish cancellation...')
            # Do not re-raise because CancelledError will be thrown into the coroutine which should
            # either re-raise it, hence we will get it in the terminal; or swallow it silently.
            # Either way, it's up to the coroutine to deal with it.

    def __str__(self):
        return 'concert-asyncio'


try:
    ip = get_ipython()
    ip.set_custom_exc(_custom_exceptions, _handler)
    # ctrl-k (abort everything, also background awaitables)
    ip.loop_runner = ConcertAsyncioRunner()
    ip.pt_app.key_bindings.add('c-k')(_abort_all_awaiting)
    ip.pt_app.key_bindings.add('c-d')(_ctrl_d)
    ip.events.register("pre_run_cell", _pre_run_cell)
except NameError as err:
    raise NameError("This module must be run after concert start") from err
