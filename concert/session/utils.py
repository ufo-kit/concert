import asyncio
import functools
import logging
import math
import os
import abc
from abc import abstractmethod
import inspect
import subprocess
import sys
import prettytable
from concert.config import AIODEBUG
from concert.coroutines.base import background, get_event_loop, run_in_loop
from concert.devices.base import Device
from concert.quantities import q


LOG = logging.getLogger(__name__)


class SubCommand(abc.ABC):

    """Base sub-command class (concert [subcommand])."""

    def __init__(self, name, opts):
        """
        SubCommand objects are loaded at run-time and injected into Concert's
        command parser.

        *name* denotes the name of the sub-command parser, e.g. "mv" for the
        MoveCommand. *opts* must be an argparse-compatible dictionary
        of command options.
        """
        self.name = name
        self.opts = opts

    @abstractmethod
    def run(self, *args, **kwargs):
        """Run the command"""
        ...


def setup_logging(name, to_stream=False, filename=None, loglevel=None):
    logformat = '[%(asctime)s] %(levelname)s: %(name)s: {}: %(message)s'
    formatter = logging.Formatter(logformat.format(name))
    logger = logging.getLogger()
    logger.setLevel(loglevel.upper())

    if to_stream:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    if filename:
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def _get_param_description_table(device, max_width):
    field_names = ["Name", "Unit", "Description"]
    widths = [int(math.floor(f * max_width)) for f in (0.3, 0.2, 0.5)]
    table = get_default_table(field_names, widths)
    table.border = False
    table.header = True

    for param in device:
        units = param.unit if hasattr(param, 'unit') else None
        row = [param.name, str(units), str(param)]
        table.add_row(row)

    return table.get_string()


def _get_param_value_table(device):
    field_names = ["Name", "Value"]
    table = get_default_table(field_names)
    table.border = False
    table.header = False

    for param in device:
        table.add_row([param.name, str(run_in_loop(param.get()))])

    if hasattr(device, 'state'):
        table.add_row(['state', device.state])

    return table.get_string()


def _current_instances(instance_type):
    # Get the second frame to skip caller who called us
    frame = inspect.stack()[2]
    instances = frame[0].f_globals

    return ((name, obj) for (name, obj)
            in list(instances.items())
            if isinstance(obj, instance_type))


def get_default_table(field_names, widths=None):
    """Return a prettytable styled for use in the shell. *field_names* is a
    list of table header strings."""
    table = prettytable.PrettyTable(field_names)
    table.border = True
    table.hrules = prettytable.ALL
    table.vertical_char = ' '
    table.junction_char = '-'

    def left_align(name):
        # Try different ways of setting the alignment to support older versions
        # of prettytable.
        try:
            table.align[name] = 'l'
        except AttributeError:
            table.set_field_align(name, 'l')

    if widths:
        for name, width in zip(field_names, widths):
            left_align(name)
            table.max_width[name] = width
    else:
        for name in field_names:
            left_align(name)

    return table


def dstate():
    """Render device state in a table."""

    field_names = ["Name", "Parameters"]
    table = get_default_table(field_names)

    for name, device in _current_instances(Device):
        values = _get_param_value_table(device)
        table.add_row([name, values])

    print(table.get_string())


def ddoc():
    """Render device documentation."""
    from concert.devices.base import Device

    n_columns = int(subprocess.check_output(['stty', 'size']).split()[1]) - 8
    field_names = ["Name", "Description", "Parameters"]
    widths = [int(math.floor(f * n_columns)) for f in (0.1, 0.2, 0.7)]
    table = get_default_table(field_names, widths)

    for name, device in _current_instances(Device):
        doc = _get_param_description_table(device, widths[2] - 4)
        table.add_row([name, inspect.getdoc(device), doc])

    print(table.get_string())


def pdoc(hide_blacklisted=True):
    """Render process documentation."""
    black_listed = ('show', 'start', 'init', 'rm', 'log', 'edit', 'import')
    field_names = ["Name", "Description"]
    table = get_default_table(field_names)

    frame = inspect.stack()[1]
    instances = frame[0].f_globals

    for name, obj in list(instances.items()):
        if not name.startswith('_') and inspect.isfunction(obj):
            if hide_blacklisted and name not in black_listed:
                table.add_row([name, inspect.getdoc(obj)])

    print(table.get_string())


def code_of(func):
    """Show implementation of *func*."""
    source = inspect.getsource(func)

    try:
        from pygments import highlight
        from pygments.lexers import PythonLexer
        from pygments.formatters import TerminalFormatter

        print(highlight(source, PythonLexer(), TerminalFormatter()))
    except ImportError:
        print(source)


def abort_awaiting(skip=None):
    """Cancel background tasks. *skip* are coroutine names which are not cancelled."""
    # Figure out if we are in a callback (ctrl-c or ctrl-k) or check_emergency_stop
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop_running = False
    else:
        loop_running = True
    LOG.log(AIODEBUG, 'async code: %s', loop_running)

    def get_task_name(task):
        return task.get_coro().__qualname__

    def get_task_directory(task):
        return os.path.dirname(task.get_coro().cr_code.co_filename)

    loop = get_event_loop()
    try:
        LOG.debug('Global abort called, loop: %d', id(loop))
    except NameError as e:
        LOG.debug("NameError `%s' in pid: %d", e, os.getpid())

    tasks = asyncio.all_tasks(loop=loop)
    LOG.log(AIODEBUG, 'Running %d tasks:\n%s', len(tasks),
            '\n'.join([get_task_name(task) for task in tasks]))

    for task in tasks:
        name = get_task_name(task)
        if skip and name in skip:
            LOG.log(AIODEBUG, 'Skipping task %s', name)
            continue
        if hasattr(task, '_is_concert_task'):
            cancelled_result = task.cancel()
            LOG.log(AIODEBUG, "Cancelling task `%s' with result %s", name, cancelled_result)


@background
async def check_emergency_stop(check, poll_interval=0.1 * q.s, exit_session=False):
    """
    check_emergency_stop(check, poll_interval=0.1*q.s, exit_session=False)
    If a callable *check* returns True abort is called. Then until it clears to False nothing is
    done and then the process begins again. *poll_interval* is the interval at which *check* is
    called. If *exit_session* is True the session exits when the emergency stop occurs.
    """
    while True:
        if check():
            LOG.error('Emergency stop')
            abort_awaiting(skip='check_emergency_stop')
            if exit_session:
                os.abort()
            while check():
                # Wait until the flag clears
                await asyncio.sleep(poll_interval.to(q.s).magnitude)
        await asyncio.sleep(poll_interval.to(q.s).magnitude)


_EXIT_FUNCTIONS = []


def register_exit_func(func):
    if func not in _EXIT_FUNCTIONS:
        _EXIT_FUNCTIONS.append(func)


def unregister_exit_func(func):
    if func in _EXIT_FUNCTIONS:
        _EXIT_FUNCTIONS.remove(func)


async def run_exit_functions():
    for func in _EXIT_FUNCTIONS:
        await func()
