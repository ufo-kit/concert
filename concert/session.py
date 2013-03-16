"""Handle session management.

A session is an ordinary Python module that is stored in a per-user
directory."""
import os
import imp
import logbook
import xdg.BaseDirectory


PATH = xdg.BaseDirectory.save_data_path('concert')

DEFAULT_LOGFILE = os.path.join(PATH, 'concert.log')


def _strip(path):
    return os.path.splitext(path)[0]


def path(session):
    """Get absolute path of *session* module."""
    return os.path.join(PATH, session + '.py')


def create(session, logfile=None):
    """Create a template with *session* name and write it

    .. note:: This will *always* overwrite session.
    """
    template = 'import quantities as q\n'
    template += '__doc__ = "This is session %s"\n' % session

    if not os.path.exists(PATH):
        os.mkdir(PATH)

    with open(path(session), 'w') as session_file:
        session_file.write(template)


def get_handler(m):
    """Return logbook handler if m.__handler__ is one, or a
    logbook.StderrHandler."""
    if hasattr(m, '__handler__'):
        if isinstance(m.__handler__, logbook.Handler):
            return m.__handler__

    return logbook.StderrHandler()


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
    return [_strip(f) for f in sessions]


def exists(session):
    """Check if *session* already exists."""
    return os.access(path(session), os.R_OK)
