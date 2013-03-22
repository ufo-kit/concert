"""
Each beamline has a certain set of motors that are used in typical experiments.
Such a group of motors can be managed with the session mechanism that is
provided by Concert and the ``concert`` command line tool. If you run the tool
without any arguments, a regular IPython shell will appear.


Three-minute tour
=================

To create a new session, use the ``init`` command with a session name::

    concert init experiment

This creates a new session *experiment*. If such a session already exists,
Concert will warn you. You can overwrite the existing session with ::

    concert init --force experiment

A session is already populated with useful imports, but you will most likely
add more definitions. For this ::

    concert edit experiment

will load the experiment session in an editor for you. Once, the session is
saved, you can start it by ::

    concert start experiment

This will load an IPython shell and import all definitions from the session
file. To remove a session, you can use the ``rm`` command::

    concert rm experiment

During an experiment, devices will output logging information. By default, this
information is gathered in a central file. To view the log for all experiments
use ::

    concert log

and to view the log for a specific experiment use ::

    concert log experiment


.. note::

    When Concert is installed system-wide, a bash completion for the
    ``concert`` tool is installed too. This means, that commands and options
    will be completed when pressing the :kbd:`Tab` key.


Customizing log output
======================

By default, logs are gathered in ``$XDG_DATA_HOME/concert/concert.log``. To
change this, you can pass the ``--logto`` and ``--logfile`` options to the
``start`` command. For example, if you want to output log to ``stderr`` use ::

    concert --logto=stderr start experiment

or if you want to get rid of any log data use ::

    concert --logto=file --logfile=/dev/null start experiment
"""
import sys
import os
import subprocess
import logbook
import concert

ARGUMENTS = {
    'edit': {'session':     {'type': str}},
    'init': {'session':     {'type': str},
             '--force':     {'action': 'store_true',
                             'help': "Overwrite existing sessions"},
             '--imports':   {'help': "Pre-import processes",
                             'metavar': 'modules',
                             'default': []}},
    'log':  {'session':     {'type': str,
                             'nargs': '?'}},
    'start': {'session':    {'type': str},
              '--logto':    {'choices': ['stderr', 'file'],
                             'default': 'file'},
              '--logfile':  {'type': str}},
    'rm':   {'sessions':    {'type': str,
                             'nargs': '+',
                             'metavar': 'session'}},
    'show': {}
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


def init(session=None, imports=[], force=False):
    """Create a new session.

    *Additional options*:

    .. cmdoption:: --force

        Create the session even if one already exists with this name.
    """
    print imports
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


def show():
    """Show available sessions."""
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
    module = concert.session.load(session)
    handler = None

    if logto == 'file':
        filename = logfile if logfile else concert.session.DEFAULT_LOGFILE
        handler = logbook.FileHandler(filename)
    else:
        handler = logbook.StderrHandler()

    handler.format_string = '[{record.time}] {record.level_name}: \
%s: {record.channel}: {record.message}' % session
    _run_shell(handler, module)


def _run_shell(handler, m=None):
    try:
        from IPython import embed
        import quantities as q

        if m:
            print m.__doc__

        banner = "Welcome to Concert {0}".format(concert.__version__)

        with handler.applicationbound():
            embed(banner1=banner)
    except ImportError as e:
        print("You must install IPython to run the Concert shell: %s" % e)


def _exit_if_not_exists(session):
    if not concert.session.exists(session):
        message = "Session `{0}' does not exist. Run `concert init {0}' first."
        print(message.format(session))
        sys.exit(1)
