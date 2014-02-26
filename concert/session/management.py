"""Handle session management.

A session is an ordinary Python module that is stored in a per-user
directory."""
import os
import sys
import imp
import shutil

_CACHED_PATH = None

_SESSION_TEMPLATE = """\"\"\"This is session {doc}\"\"\"

import logging
import concert
concert.require("{version}")

from concert.quantities import q
from concert.session.utils import ddoc, dstate, pdoc, code_of

LOG = logging.getLogger(__name__)
"""


def path(session=None):
    """
    Get absolute path of *session* module or base path if *session* is None.
    """
    global _CACHED_PATH

    if _CACHED_PATH is None:
        env = os.environ
        if "VIRTUAL_ENV" in env:
            env["XDG_DATA_HOME"] = os.path.join(env["VIRTUAL_ENV"], "share")

        import xdg.BaseDirectory
        _CACHED_PATH = xdg.BaseDirectory.save_data_path('concert')

    if session is None:
        return _CACHED_PATH

    return os.path.join(_CACHED_PATH, session + '.py')


def logfile_path():
    return os.path.join(path(), 'concert.log')


def create(session, imports=()):
    """Create a template with *session* name and write it.

    For each name in *imports* try to load it and insert `from
    concert.processes.name import *` into the session file.

    .. note:: This will *always* overwrite session.
    """
    from concert import get_canonical_version
    template = _SESSION_TEMPLATE.format(version=get_canonical_version(), doc=session)

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

    if not os.path.exists(path()):
        os.mkdir(path())

    with open(path(session), 'w') as session_file:
        session_file.write(template)


def remove(session):
    """Remove a *session*."""
    if exists(session):
        os.unlink(path(session))


def move(source, target):
    """Move *source* to *target*."""
    os.rename(path(source), path(target))


def copy(source, target):
    """Copy *source* to *target*."""
    shutil.copy(path(source), path(target))


def load(session):
    """Load *session* and return the module."""
    return imp.load_source(session, path(session))


def get_existing():
    """Get all existing session names."""
    sessions = [f for f in os.listdir(path()) if f.endswith('.py')]
    return [os.path.splitext(f)[0] for f in sessions]


def exists(session):
    """Check if *session* already exists."""
    return os.access(path(session), os.R_OK)


def exit_if_not_exists(session):
    """Exit if *session* does not exist with a message."""
    if not exists(session):
        message = "Session `{0}' does not exist. Run `concert init {0}' first."
        sys.exit(message.format(session))
