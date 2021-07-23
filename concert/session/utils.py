import logging
import math
import os
import inspect
import subprocess
import time
import prettytable
from concert.casync import threaded, wait
from concert.devices.base import abort as device_abort
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
        table.add_row([param.name, str(param.get().result())])

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


def cdoc():
    """Render device documentation."""
    from concert.commands import COMMANDS

    field_names = ["Name", "Description"]
    table = get_default_table(field_names)

    for name in sorted(COMMANDS):
        table.add_row([name, inspect.getdoc(COMMANDS[name])])

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


def abort():
    """Abort all actions related with parameters on all devices."""
    from concert.devices.base import Device
    from concert.experiments.base import Acquisition, Experiment

    # First abort experiments
    futures = [ex.abort() for (name, ex) in _current_instances(Experiment)]
    # Then acquisitions, in case there are some standalone ones
    futures += [acq.abort() for (name, acq) in _current_instances(Acquisition)]
    futures += device_abort((device for (name, device) in _current_instances(Device)))

    return futures


@threaded
def check_emergency_stop(check, poll_interval=0.1*q.s, exit_session=False):
    """
    check_emergency_stop(check, poll_interval=0.1*q.s, exit_session=False)
    If a callable *check* returns True abort is called. Then until it clears to False nothing is
    done and then the process begins again. *poll_interval* is the interval at which *check* is
    called. If *exit_session* is True the session exits when the emergency stop occurs.
    """
    while True:
        if check():
            futures = abort()
            LOG.error('Emergency stop')
            wait(futures)
            if exit_session:
                os.abort()
            while check():
                # Wait until the flag clears
                time.sleep(poll_interval.to(q.s).magnitude)
        time.sleep(poll_interval.to(q.s).magnitude)
