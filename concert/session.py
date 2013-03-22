"""Handle session management.

A session is an ordinary Python module that is stored in a per-user
directory."""
import os
import imp
import xdg.BaseDirectory


PATH = xdg.BaseDirectory.save_data_path('concert')

DEFAULT_LOGFILE = os.path.join(PATH, 'concert.log')


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
