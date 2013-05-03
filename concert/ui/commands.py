"""
Command functions used by the ``concert`` command-line interface tool.
"""
from __future__ import print_function

import sys
import os
import subprocess
import logbook
import traceback
import concert
from concert.base import (UnitError,
                          LimitError,
                          ParameterError,
                          ReadAccessError,
                          WriteAccessError)

ARGUMENTS = {
    'edit': {'session': {'type': str}},
    'init': {'session': {'type': str},
             '--force': {'action': 'store_true',
                         'help': "Overwrite existing sessions"},
             '--imports': {'help': "Pre-import processes",
                           'metavar': 'modules',
                           'default': ''}},
    'log': {'session': {'type': str,
                        'nargs': '?'}},
    'start': {'session': {'type': str},
              '--logto': {'choices': ['stderr', 'file'],
                          'default': 'file'},
              '--logfile': {'type': str}},
    'rm': {'sessions': {'type': str,
                        'nargs': '+',
                        'metavar': 'session'}},
    'show': {'session': {'type': str,
                         'nargs': '?',
                         'default': None,
                         'help': "Show details"}}
}


def edit(session=None):
    """Edit a session.

    Edit the session file by launching ``$EDITOR`` with the associated Python
    module file. This file can contain any kind of Python code, but you will
    most likely just add device definitions such as this::

        from concert.devices.axes.crio import LinearAxis

        crio1 = LinearAxis(None)
    """
    _exit_if_not_exists(session)
    env = os.environ
    editor = env['EDITOR'] if 'EDITOR' in env else 'vi'
    subprocess.call([editor, concert.session.path(session)])


def init(session=None, imports="", force=False):
    """Create a new session.

    *Additional options*:

    .. cmdoption:: --force

        Create the session even if one already exists with this name.
    """
    if concert.session.exists(session) and not force:
        message = "Session `{0}' already exists."
        message += " Use --force to create it anyway."
        print(message.format(session))
    else:
        concert.session.create(session, imports.split())


def log(session=None):
    """Show session logs.

    If a *session* is not given, the log command shows entries from all
    sessions.
    """
    logfile = concert.session.DEFAULT_LOGFILE

    if not os.path.exists(logfile):
        return

    # This is danger zone here because we run subprocess.call with shell=True.
    # However, the only input that we input is args.session which we check
    # first and the logfile itself.

    if session:
        _exit_if_not_exists(session)
        cmd = 'grep "{0}:" {1} | less'.format(session, logfile)
    else:
        cmd = 'less {0}'.format(logfile)

    subprocess.call(cmd, shell=True)


def rm(sessions=[]):
    """Remove one or more sessions.

    .. note::

        Be careful. The session file is unlinked from the file system and no
        backup is made.

    """
    for session in sessions:
        print("Removing {0}...".format(session))
        concert.session.remove(session)


def show(session=None):
    """Show available sessions or details of a given *session*."""
    if session:
        try:
            module = concert.session.load(session)
            print(module.__doc__)
        except IOError:
            print("Cannot find {0}".format(session))
        except ImportError as exception:
            print("Cannot import {0}: {1}".format(session, exception))
    else:
        sessions = concert.session.get_existing()
        print("Available sessions:")

        for session in sessions:
            print("  %s" % session)


def start(session=None, logto='file', logfile=None):
    """Start a session.

    Load the session file and launch an IPython shell. Every definition that
    was made in the module file is available via the ``m`` variable. Moreover,
    the quantities package is already loaded and named ``q``. So, once the
    session has started you could access motors like this::

        $ concert start tomo

        This is session tomo
        Welcome to Concert 0.0.1
        In [1]: m.crio1.set_positon(2.23 * q.mm)
        In [2]: m.crio1.get_position()
        Out[2]: array(2.23) * mm

    *Additional options*:

    .. cmdoption:: --logto={stderr, file}

        Specify a method for logging events. If this flag is not specified,
        ``file`` is used and assumed to be
        ``$XDG_DATA_HOME/concert/concert.log``.

    .. cmdoption:: --logfile=<filename>

        Specify a log file if ``--logto`` is set to ``file``.

    """
    _exit_if_not_exists(session)
    handler = None

    if logto == 'file':
        filename = logfile if logfile else concert.session.DEFAULT_LOGFILE
        handler = logbook.FileHandler(filename)
    else:
        handler = logbook.StderrHandler()

    handler.format_string = '[{record.time}] {record.level_name}: \
%s: {record.channel}: {record.message}' % session

    # Add session path, so that sessions can import other sessions
    sys.path.append(concert.session.PATH)
    try:
        module = concert.session.load(session)
    except Exception as exception:
        traceback.print_exc()
        sys.exit(1)

    _run_shell(handler, module)


def _get_module_variables(module):
    attrs = [attr for attr in dir(module) if not attr.startswith('_')]
    return dict((attr, getattr(module, attr)) for attr in attrs)


def _run_shell(handler, module=None):
    def _handler(shell, etype, evalue, traceback, tb_offset=None):
        print("Sorry, {0}".format(str(evalue)))
        return None

    try:
        from IPython.frontend.terminal.embed import InteractiveShellEmbed
        import quantities as q

        print("Welcome to Concert {0}".format(concert.__version__))

        if module:
            print(module.__doc__)

        globals().update(_get_module_variables(module))

        with handler.applicationbound():
            shell = InteractiveShellEmbed(banner1='')

            exceptions = (UnitError,
                          LimitError,
                          ParameterError,
                          ReadAccessError,
                          WriteAccessError)
            shell.set_custom_exc(exceptions, _handler)
            shell()
    except ImportError as exception:
        msg = "You must install IPython to run the Concert shell: {0}"
        print(msg.format(exception))


def _exit_if_not_exists(session):
    if not concert.session.exists(session):
        message = "Session `{0}' does not exist. Run `concert init {0}' first."
        print(message.format(session))
        sys.exit(1)
