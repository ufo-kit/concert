"""Handle session management.

A session is an ordinary Python module that is stored in a per-user
directory."""
import os
import imp
import prettytable
from inspect import getdoc
from concert.ui import get_default_table


def _get_save_data_path():
    env = os.environ
    if "VIRTUAL_ENV" in env:
        env["XDG_DATA_HOME"] = os.path.join(env["VIRTUAL_ENV"], "share")
    
    import xdg.BaseDirectory
    return xdg.BaseDirectory.save_data_path('concert')


PATH = _get_save_data_path()
DEFAULT_LOGFILE = os.path.join(PATH, 'concert.log')


def _get_param_description_table(motor):
    field_names = ["Name", "Access", "Unit", "Description"]
    table = get_default_table(field_names)
    table.border = False
    table.header = True

    def access_nick(parameter):
        result = 'r' if parameter.is_readable() else ''
        result += 'w' if parameter.is_writable() else ''
        return result

    for param in motor:
        dims = param.unit.dimensionality.string if param.unit else None
        row = [param.name, access_nick(param), str(dims), getdoc(param)]
        table.add_row(row)

    return table.get_string()


def _get_param_value_table(motor):
    field_names = ["Name", "Value"]
    table = get_default_table(field_names)
    table.border = False
    table.header = False

    for param in motor:
        if param.is_readable():
            table.add_row([param.name, str(param.get().result())])

    return table.get_string()


class DeviceDocumentation(list):
    """Render device documentation."""
    def __repr__(self):
        field_names = ["Name", "Description", "Parameters"]
        table = get_default_table(field_names)

        for device in self:
            doc = _get_param_description_table(device)
            table.add_row([device.__class__.__name__, getdoc(device), doc])

        return table.get_string()


class ProcessDocumentation(list):
    """Render process documentation."""
    def __repr__(self):
        field_names = ["Name", "Description"]
        table = get_default_table(field_names)

        for process in self:
            table.add_row([process.__name__, getdoc(process)])

        return table.get_string()


class DeviceState(list):
    """Render device state in a table."""
    def __repr__(self):
        field_names = ["Name", "Parameters"]
        table = get_default_table(field_names)

        for device in self:
            values = _get_param_value_table(device)
            table.add_row([device.__class__.__name__, values])

        return table.get_string()


def path(session):
    """Get absolute path of *session* module."""
    return os.path.join(PATH, session + '.py')


def create(session, imports=[]):
    """Create a template with *session* name and write it.

    For each name in *imports* try to load it and insert `from
    concert.processes.name import *` into the session file.

    .. note:: This will *always* overwrite session.
    """
    template = 'import quantities as q\n'
    template += '\n'
    template += 'from concert.session import ddoc\n'
    template += 'from concert.session import dstate\n'
    template += 'from concert.session import pdoc\n'
    template += '\n'
    template += '__doc__ = "This is session %s"\n' % session

    def _module_exists(module_name):
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False

    for module in imports:
        module_name = 'concert.processes.{0}'.format(module)

        if _module_exists(module_name):
            template += 'from {0} import *'.format(module_name)
        else:
            print("{0} not found.".format(module_name))

    if not os.path.exists(PATH):
        os.mkdir(PATH)

    with open(path(session), 'w') as session_file:
        session_file.write(template)


def remove(session):
    """Remove a *session*."""
    if exists(session):
        os.unlink(path(session))


def load(session):
    """Load *session* and return the module."""
    return imp.load_source('m', path(session))


def get_existing():
    """Get all existing session names."""
    sessions = [f for f in os.listdir(PATH) if f.endswith('.py')]
    return [os.path.splitext(f)[0] for f in sessions]


def exists(session):
    """Check if *session* already exists."""
    return os.access(path(session), os.R_OK)


ddoc = DeviceDocumentation()
pdoc = ProcessDocumentation()
dstate = DeviceState()
