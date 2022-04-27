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
from concert.session.utils import abort_awaiting


LOG = logging.getLogger(__name__)


def _handler(_shell, _etype, evalue, _traceback_, tb_offset=None):
    if _etype == asyncio.CancelledError:
        # Silently log the error message without polluting the terminal
        messages = traceback.extract_tb(_traceback_).format()
        if len(messages) > 1:
            messages = messages[1:]
        LOG.info('asyncio.CancelledError\n' + ''.join(messages))
    elif _etype == KeyboardInterrupt:
        # Silently log the error message without polluting the terminal
        messages = traceback.extract_tb(_traceback_).format()
        if len(messages) > 2:
            # Skip IPython part and our sigint handler
            messages = messages[1:-1]
        LOG.info('KeyboardInterrupt in normal mode\n' + ''.join(messages))
    else:
        print("Sorry, {0}".format(str(evalue)))
    return None


def _abort_all_awaiting(event):
    abort_awaiting(background=True)


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

try:
    ip = get_ipython()
    ip.set_custom_exc(_custom_exceptions, _handler)
    # ctrl-k (abort everything, also background awaitables)
    ip.pt_app.key_bindings.add('c-k')(_abort_all_awaiting)
except NameError as err:
    raise NameError("This module must be run after concert start") from err
