"""Handle session management.

A session is an ordinary Python module that is stored in a per-user
directory."""
import os
import imp
import xdg.BaseDirectory


PATH = xdg.BaseDirectory.save_data_path('concert')


def _strip(path):
    return os.path.splitext(path)[0]


def path(session):
    """Get absolute path of *session* module."""
    return os.path.join(PATH, session + '.py')


def create(session):
    """Create a template with *session* name and write it

    .. note:: This will *always* overwrite session.
    """
    template = r"""__doc__ = "This is session %s" """ % session

    if not os.path.exists(PATH):
        os.mkdir(PATH)

    with open(path(session), 'w') as fp:
        fp.write(template)


def load(session):
    return imp.load_source('m', path(session))


def get_existing():
    """Get all existing session names."""
    sessions = [f for f in os.listdir(PATH) if f.endswith('.py')]
    return [_strip(f) for f in sessions]


def exists(session):
    """Check if *session* already exists."""
    return os.access(path(session), os.R_OK)
