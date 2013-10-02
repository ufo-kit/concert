"""
Command functions used by the ``concert`` command-line interface tool.
"""
from __future__ import print_function

import sys
import os
import re
import subprocess
import tempfile
import shutil
import logbook
import traceback
import contextlib
import urlparse
import urllib2
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
    'fetch': {'url': {'type': str,
                      'help': "Fetch a Python module and save as a session."
                              " Note: Server certificates of HTTPS requests"
                              " are NOT verified!"},
              '--force': {'action': 'store_true',
                          'help': "Overwrite existing sessions"},
              '--repo': {'action': 'store_true',
                         'help':
                         "Checkout Git repository and import all files"}},
    'log': {'session': {'type': str,
                        'nargs': '?'}},
    'start': {'session': {'type': str},
              '--logto': {'choices': ['stderr', 'file'],
                          'default': 'file'},
              '--logfile': {'type': str}},
    'rm': {'sessions': {'type': str,
                        'nargs': '+',
                        'metavar': 'session'}},
    'mv': {'source': {'type': str,
                      'help': "Name of the source session"},
           'target': {'type': str,
                      'help': "Name of the target session"}},
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


def _get_url(path_or_url):
    result = urlparse.urlsplit(path_or_url)

    if result.scheme:
        return path_or_url

    if not os.path.exists(path_or_url):
        sys.exit("Cannot find module `{0}'.".format(path_or_url))

    result = ('file', '', os.path.abspath(path_or_url), '', '')
    return urlparse.urlunsplit(result)


def _fetch_file(url, force):
    if not url.endswith('.py'):
        sys.exit("`{0}' is not a Python module".format(url))

    session_name = os.path.basename(url[:-3])

    if concert.session.exists(session_name) and not force:
        sys.exit("`{0}' already exists".format(session_name))

    local_url = _get_url(url)

    with contextlib.closing(urllib2.urlopen(local_url)) as data:
        save_path = os.path.join(concert.session.PATH, session_name + '.py')
        with open(save_path, 'w') as output:
            output.write(data.read())


def _fetch_repo(url, force):
    path = tempfile.mkdtemp()
    cmd = 'git clone --quiet {0} {1}'.format(url, path)
    proc = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    out, err = proc.communicate()

    if proc.returncode != 0:
        sys.exit("Could not clone {0}.".format(url))

    for filename in (x for x in os.listdir(path) if x.endswith('.py')):
        session_name = os.path.basename(filename[:-3])

        if concert.session.exists(session_name) and not force:
            print("`{0}' already exists (use --force to install"
                  " anyway)".format(session_name))
        else:
            print("Add session {0} ...".format(filename[:-3]))
            shutil.copy(os.path.join(path, filename), concert.session.PATH)

    shutil.rmtree(path)


def fetch(url, force=False, repo=False):
    """Import an existing *session*."""
    if repo:
        _fetch_repo(url, force)
    else:
        _fetch_file(url, force)


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


def mv(source, target):
    """Move session *source* to *target*."""
    if not concert.session.exists(source):
        sys.exit("`{}' does not exist".format(source))

    if concert.session.exists(target):
        sys.exit("`{}' already exists".format(target))

    concert.session.move(source, target)
    print("Renamed {} -> {}".format(source, target))


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
    except:
        traceback.print_exc()
        sys.exit(1)

    _run_shell(handler, module)


def _get_module_variables(module):
    attrs = [attr for attr in dir(module) if not attr.startswith('_')]
    return dict((attr, getattr(module, attr)) for attr in attrs)


def _compare_versions(v1, v2):
    """Compare two version numbers and return cmp compatible result"""
    def normalize(v):
        return [int(x) for x in re.sub(r'(\.0+)*$', '', v).split(".")]

    n1 = normalize(v1)
    n2 = normalize(v2)
    return (n1 > n2) - (n1 < n2)


def _run_shell(handler, module=None):
    def _handler(_shell, _etype, evalue, _traceback_, tb_offset=None):
        print("Sorry, {0}".format(str(evalue)))
        return None

    from concert.quantities import q

    print("Welcome to Concert {0}".format(concert.__version__))

    if module:
        print(module.__doc__)

    globals().update(_get_module_variables(module))

    try:
        with handler.applicationbound():
            import IPython

            version = IPython.__version__

            # Jeez, let's see what comes next ...
            if _compare_versions(version, '0.11') < 0:
                from IPython.Shell import IPShellEmbed
                shell = IPShellEmbed()
            elif _compare_versions(version, '1.0') < 0:
                from IPython.frontend.terminal.embed import \
                    InteractiveShellEmbed
                shell = InteractiveShellEmbed(banner1='')
            else:
                from IPython.terminal.embed import InteractiveShellEmbed
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
        sys.exit(message.format(session))
