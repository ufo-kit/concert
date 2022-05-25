import asyncio
import logging
import math
import os
import inspect
import subprocess
import prettytable
from concert.config import AIODEBUG
from concert.coroutines.base import background, get_event_loop, run_in_loop
from concert.devices.base import Device
from concert.quantities import q


LOG = logging.getLogger(__name__)


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


def abort_awaiting(background=False, skip=None):
    """Abort task currently being awaited in the session. Return True if there is a task being
    awaited, otherwise False. This function does not touch tasks running in the background unless
    *background* is True, in which case it cancels all awaitables.
    """
    # Figure out if we are in a callback (ctrl-c or ctrl-k) or check_emergency_stop
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop_running = False
    else:
        loop_running = True
    LOG.log(AIODEBUG, 'async code: %s', loop_running)

    def get_task_name(task):
        # _coro instead of get_coro() for Python 3.7 compatibility
        return task._coro.__qualname__

    def get_task_directory(task):
        # _coro instead of get_coro() for Python 3.7 compatibility
        return os.path.dirname(task._coro.cr_code.co_filename)

    loop = get_event_loop()
    try:
        LOG.debug('Global abort called, loop: %d, IPython loop: %d', id(loop),
                  id(get_ipython().pt_loop))
    except NameError:
        LOG.debug('NameError in pid: %d', os.getpid())

    tasks = asyncio.all_tasks(loop=loop)
    LOG.log(AIODEBUG, 'Running %d tasks:\n%s', len(tasks),
            '\n'.join([get_task_name(task) for task in tasks]))

    for task in tasks:
        name = get_task_name(task)
        if skip and name in skip:
            LOG.log(AIODEBUG, 'Skipping task %s', name)
            continue
        abortable = False
        if background and hasattr(task, '_is_concert_task'):
            # ctrl-k, cancel everything from us
            abortable = True
        elif 'run_cell_async' in name:
            # ctrl-c
            # TODO: make this cleaner and robust (run_cell_async may change its name and so on)
            abortable = True
        if abortable:
            # We either abort everything which is enabled for aborting or the current coroutine
            # which is being awaited in the session
            cancelled_result = task.cancel()
            LOG.log(AIODEBUG, "Cancelling task `%s' with result %s", name, cancelled_result)
            if not background:
                return True

    if background:
        # Either ctrl-k or ctrl-c in a non-async function
        # Use ipython magic because _current_instances won't work here (different stack)
        try:
            uns = get_ipython().user_ns
        except NameError:
            # Invoked from tests
            return False
        devices = [(name, value) for name, value in uns.items() if isinstance(value, Device)]
        emergency_stops = []
        emergency_device_names = []
        for name, value in uns.items():
            if not name.startswith('_') and isinstance(value, Device):
                # ipython adds _1, _2, ... variables to user_ns when people type variable names in
                # the shell
                LOG.info("Emergency stop on `%s'", name)
                emergency_stops.append(value.emergency_stop())
                emergency_device_names.append(name)

    return False


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
            abort_awaiting(background=True, skip='check_emergency_stop')
            if exit_session:
                os.abort()
            while check():
                # Wait until the flag clears
                await asyncio.sleep(poll_interval.to(q.s).magnitude)
        await asyncio.sleep(poll_interval.to(q.s).magnitude)
