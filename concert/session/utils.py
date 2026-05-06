import asyncio
import logging
import math
import os
import abc
from abc import abstractmethod
import inspect
import subprocess
import atexit
import contextlib
import sys
import shutil
import subprocess
import tempfile
import zipfile
import prettytable
import concert
from concert.config import AIODEBUG
from concert.coroutines.base import background, get_event_loop, run_in_loop
from concert._aimport import eval_source, register
import concert.session.management as cs
from concert.devices.base import Device
from concert.quantities import q

LOG = logging.getLogger(__name__)


class SubCommand(abc.ABC):
    """Base sub-command class (concert [subcommand])."""

    def __init__(self, name, opts):
        """
        SubCommand objects are loaded at run-time and injected into Concert's
        command parser.

        *name* denotes the name of the sub-command parser, e.g. "mv" for the
        MoveCommand. *opts* must be an argparse-compatible dictionary
        of command options.
        """
        self.name = name
        self.opts = opts

    @abstractmethod
    def run(self, *args, **kwargs):
        """Run the command"""
        ...


def setup_logging(name, to_stream=False, filename=None, loglevel=None):
    logformat = '[%(asctime)s] %(levelname)s: %(name)s: {}: %(message)s'
    formatter = logging.Formatter(logformat.format(name))
    logger = logging.getLogger()
    logger.setLevel(loglevel.upper())

    if to_stream:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    if filename:
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


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


def abort_awaiting(skip=None):
    """Cancel background tasks. *skip* are coroutine names which are not cancelled."""
    # Figure out if we are in a callback (ctrl-c or ctrl-k) or check_emergency_stop
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop_running = False
    else:
        loop_running = True
    LOG.log(AIODEBUG, 'async code: %s', loop_running)

    def get_task_name(task):
        return task.get_coro().__qualname__

    def get_task_directory(task):
        return os.path.dirname(task.get_coro().cr_code.co_filename)

    loop = get_event_loop()
    try:
        LOG.debug('Global abort called, loop: %d', id(loop))
    except NameError as e:
        LOG.debug("NameError `%s' in pid: %d", e, os.getpid())

    tasks = asyncio.all_tasks(loop=loop)
    LOG.log(AIODEBUG, 'Running %d tasks:\n%s', len(tasks),
            '\n'.join([get_task_name(task) for task in tasks]))

    for task in tasks:
        name = get_task_name(task)
        if skip and name in skip:
            LOG.log(AIODEBUG, 'Skipping task %s', name)
            continue
        if hasattr(task, '_is_concert_task'):
            cancelled_result = task.cancel()
            LOG.log(AIODEBUG, "Cancelling task `%s' with result %s", name, cancelled_result)


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
            abort_awaiting(skip='check_emergency_stop')
            if exit_session:
                os.abort()
            while check():
                # Wait until the flag clears
                await asyncio.sleep(poll_interval.to(q.s).magnitude)
        await asyncio.sleep(poll_interval.to(q.s).magnitude)


STARTUP_FMT = """
try:
    from {0} import *
except Exception as e:
    import os
    import signal
    import sys
    import traceback
    print("\\nAn error occured while starting session `{0}':")
    print("-------------------------------------------" + "-" * len('{0}'))
    print(traceback.format_exc(), file=sys.stderr)
    if os.path.exists(r"{1}"):
        os.remove(r"{1}")
        print(r"Removed lock file {1}")
    os.kill(os.getpid(), signal.SIGTERM)
"""


def delete_lock_file(lockfile):
    if os.path.exists(lockfile):
        os.remove(lockfile)
        LOG.debug("Removed lock file %s", lockfile)


def get_prompt_config(path):
    """Customize the prompt for session in *path*."""
    from IPython.terminal.prompts import Prompts, Token

    session_name = os.path.splitext(os.path.basename(path))[0]

    class MyPrompt(Prompts):
        def in_prompt_tokens(self, cli=None):
            return [(Token, session_name), (Token.Prompt, ' > ')]

        def out_prompt_tokens(self):
            return []

    return MyPrompt


class InitCommand(SubCommand):
    """Create a new session."""

    def __init__(self):
        opts = {'session': {'type': str},
                '--force': {'action': 'store_true',
                            'help': "Overwrite existing sessions"},
                '--imports': {'help': "Pre-import processes",
                              'metavar': 'modules',
                              'default': ''}}
        super(InitCommand, self).__init__('init', opts)

    def run(self, session=None, imports="", force=False):
        if cs.exists(session) and not force:
            message = "Session `{0}' already exists."
            message += " Use --force to create it anyway."
            print(message.format(session))
        else:
            cs.create(session, imports.split())


class EditCommand(SubCommand):
    """Edit a session."""

    def __init__(self):
        opts = {'session': {'type': str}}
        super(EditCommand, self).__init__('edit', opts)

    def run(self, session=None):
        if not cs.exists(session):
            print("Session not found, creating {}.".format(session))
            InitCommand().run(session)

        env = os.environ
        editor = env['EDITOR'] if 'EDITOR' in env else 'vi'
        subprocess.call([editor, cs.path(session)])


class LogCommand(SubCommand):
    """Show session logs."""

    def __init__(self):
        opts = {'session': {'type': str,
                            'nargs': '?'},
                '--follow': {'action': 'store_true',
                             'help': 'Show current log'}}
        super(LogCommand, self).__init__('log', opts)

    def run(self, session=None, follow=False):
        logfile = cs.logfile_path()

        if not os.path.exists(logfile):
            return

        # This is danger zone here because we run subprocess.call with
        # shell=True.  However, the only input that we input is args.session
        # which we check first and the logfile itself.

        if session:
            cs.exit_if_not_exists(session)

            if follow:
                cmd = 'tail -f {} | grep --line-buffered "{}:"'.format(logfile, session)
            else:
                cmd = 'grep "{0}:" {1} | less'.format(session, logfile)
        else:
            if follow:
                cmd = 'tail -f {}'.format(logfile)
            else:
                cmd = 'less {}'.format(logfile)

        try:
            subprocess.call(cmd, shell=True)
        except KeyboardInterrupt:
            # When following we can only leave tail by C-c, hence to avoid
            # spamming the terminal with a stack trace we just ignore the
            # Keyboardinterrupt exception.
            pass


class ShowCommand(SubCommand):
    """Show available sessions or details of a given *session*."""

    def __init__(self):
        opts = {'session': {'type': str,
                            'nargs': '?',
                            'default': None,
                            'help': "Show details"}}
        super(ShowCommand, self).__init__('show', opts)

    def run(self, session=None):
        if session:
            try:
                docstring = cs.get_docstring(session)
                print(docstring)
            except IOError:
                print("Cannot find {0}".format(session))
            except ImportError as exception:
                print("Cannot import {0}: {1}".format(session, exception))
        else:
            sessions = cs.get_existing()
            print("Available sessions:")

            for session in sessions:
                print("  %s" % session)


class MoveCommand(SubCommand):
    """Move session *source* to *target*."""

    def __init__(self):
        opts = {'source': {'type': str,
                           'help': "Name of the source session"},
                'target': {'type': str,
                           'help': "Name of the target session"}}
        super(MoveCommand, self).__init__('mv', opts)

    def run(self, source, target):
        if not cs.exists(source):
            sys.exit("`{}' does not exist".format(source))

        if cs.exists(target):
            sys.exit("`{}' already exists".format(target))

        cs.move(source, target)
        print("Renamed {} -> {}".format(source, target))


class CopyCommand(SubCommand):
    """Copy session *source* to *target*."""

    def __init__(self):
        opts = {'source': {'type': str,
                           'help': "Name of the source session"},
                'target': {'type': str,
                           'help': "Name of the target session"}}
        super(CopyCommand, self).__init__('cp', opts)

    def run(self, source, target):
        if not cs.exists(source):
            sys.exit("`{}' does not exist".format(source))

        if cs.exists(target):
            sys.exit("`{}' already exists".format(target))

        cs.copy(source, target)
        print("Copied {} -> {}".format(source, target))


class RemoveCommand(SubCommand):
    """Remove one or more sessions."""

    def __init__(self):
        opts = {'sessions': {'type': str,
                             'nargs': '+',
                             'metavar': 'session'}}
        super(RemoveCommand, self).__init__('rm', opts)

    def run(self, sessions=[]):
        for session in sessions:
            print("Removing {0}...".format(session))
            cs.remove(session)


class ImportCommand(SubCommand):
    """Import an existing *session*."""

    def __init__(self):
        opts = {'url': {'nargs': '+', 'type': str,
                        'help': "Import a Python module and save as a session."
                                " Note: Server certificates of HTTPS requests"
                                " are NOT verified!"},
                '--force': {'action': 'store_true',
                            'help': "Overwrite existing sessions"},
                '--repo': {'action': 'store_true',
                           'help':
                               "Checkout Git repository and import all files"}}
        super(ImportCommand, self).__init__('import', opts)

    def run(self, url, force=False, repo=False):
        for u in url:
            if repo:
                self._import_repo(u, force)
            else:
                self._import_file(u, force)

    def _import_repo(self, url, force):
        path = tempfile.mkdtemp()
        cmd = 'git clone --quiet {0} {1}'.format(url, path)
        proc = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        out, err = proc.communicate()

        if proc.returncode != 0:
            sys.exit("Could not clone {0}.".format(url))

        for filename in (x for x in os.listdir(path) if x.endswith('.py')):
            session_name = os.path.basename(filename[:-3])

            if cs.exists(session_name) and not force:
                print("`{0}' already exists (use --force to install"
                      " anyway)".format(session_name))
            else:
                print("Add session {0} ...".format(filename[:-3]))
                shutil.copy(os.path.join(path, filename),
                            cs.path())

        shutil.rmtree(path)

    def _import_file(self, url, force):
        import urllib.request
        import urllib.error
        import urllib.parse

        if not url.endswith('.py'):
            sys.exit("`{0}' is not a Python module".format(url))

        session_name = os.path.basename(url[:-3])

        if cs.exists(session_name) and not force:
            sys.exit("`{0}' already exists".format(session_name))

        print("Add session {0} ...".format(session_name))
        local_url = self._get_url(url)

        with contextlib.closing(urllib.request.urlopen(local_url)) as data:
            with open(cs.path(session_name), 'w') as output:
                output.write(data.read())

    def _get_url(self, path_or_url):
        import urllib.parse

        result = urllib.parse.urlsplit(path_or_url)

        if result.scheme:
            return path_or_url

        if not os.path.exists(path_or_url):
            sys.exit("Cannot find module `{0}'.".format(path_or_url))

        result = ('file', '', os.path.abspath(path_or_url), '', '')
        return urllib.parse.urlunsplit(result)


class ExportCommand(SubCommand):
    """Export all sessions as a Zip archive."""

    def __init__(self):
        opts = {'name': {'type': str,
                         'help': "Name of the archive"}}
        super(ExportCommand, self).__init__('export', opts)

    def run(self, name):
        name = name if name.endswith('.zip') else name + '.zip'

        with zipfile.ZipFile(name, 'w') as archive:
            for path in (cs.path(session) for session in cs.get_existing()):
                archive.writestr(os.path.basename(path), open(path).read())


class StartCommand(SubCommand):
    """Start a session."""

    def __init__(self):
        opts = {'session': {'nargs': '?', 'type': str, 'default': None},
                '--filename': {'type': str, 'default': None},
                '--logto': {'choices': ['stderr', 'file'],
                            'default': 'file'},
                '--logfile': {'type': str},
                '--loglevel': {'choices': ['perfdebug', 'aiodebug', 'debug', 'info', 'warning',
                                           'error', 'critical'],
                               'default': 'info'},
                '--non-interactive': {'action': 'store_true'}}
        super(StartCommand, self).__init__('start', opts)

    def run(self, session=None, filename=None,
            non_interactive=False,
            logto='file', logfile=None, loglevel=None):
        import IPython

        if IPython.version_info >= (8, 0):
            # Ipython creates a new event loop in get_asyncio_loop function, but only sets it in
            # the prompt_for_code of the
            # IPython.terminal.interactiveshell.TerminalInteractiveShell, so when we use
            # enabel_gui programatically, that loop setting will never be triggered. Thus, we do
            # it ourselves before anything else with any concert part happens.
            from IPython.core.async_helpers import get_asyncio_loop
            asyncio.set_event_loop(get_asyncio_loop())

        # From now on, we can import code with `await' outside of `async def' functions
        # It is crucial to call this here, after setting the asyncio's loop to the one from IPython,
        # otherwise we might be importing with another event loop than the one used in the concert
        # session and thus have devices and Locks and other things bound to a wrong loop.
        register(paths=[cs.path()])

        if session:
            cs.exit_if_not_exists(session)

        logfilename = (logfile if logfile else cs.logfile_path()) if logto == 'file' else None
        setup_logging(session, to_stream=logto == 'stderr', filename=logfilename, loglevel=loglevel)
        # Add session path, so that sessions can import other sessions
        sys.path.append(cs.path())
        path = filename or cs.path(session)

        if path and path.endswith('.py'):
            lockdir = os.path.join(os.path.expanduser("~"), ".concert")
            os.makedirs(lockdir, mode=0o755, exist_ok=True)
            lockfile = os.path.join(lockdir, '.' + os.path.basename(path))
            if not cs.is_multiinstance(session):
                if os.path.exists(lockfile):
                    print(f"An instance of `{session}' seems to be already running.")
                    print(
                        f"If you are sure there are no running instances, delete `{lockfile}' "
                        "and start again."
                    )
                    raise SessionLockException(session, lockfile)
                with open(lockfile, 'w'):
                    pass
                atexit.register(delete_lock_file, lockfile)

            locked_devices = []
            for device_lock in cs.get_device_locks(session):
                lockfile = os.path.join(lockdir, '.device_lock_' + device_lock)
                if os.path.exists(lockfile):
                    print(f"A device lock `{device_lock}' seems to be active.")
                    print(
                        f"If you are sure there are no running instances, delete `{lockfile}' "
                        "and start again."
                    )
                    locked_devices.append(device_lock)
                if locked_devices:
                    raise DeviceLockException(locked_devices, lockfile)

                with open(lockfile, 'w'):
                    pass
                atexit.register(delete_lock_file, lockfile)


        if non_interactive:
            with open(path, "rb") as f:
                eval_source(f.read(), {}, filename=path)
        else:
            self.run_shell(path=path, session=session, lockfile=lockfile)

    def run_shell(self, path=None, session=None, lockfile=None):
        import IPython
        import traitlets.config

        print("Welcome to Concert {0}".format(concert.__version__))

        ip_config = traitlets.config.Config()
        if path and path.endswith('.py'):

            docstring = cs.get_docstring(path)
            if docstring:
                print(docstring)

            if session is None:
                # --filename must have been specified
                session = os.path.splitext(os.path.basename(path))[0]
                sys.path.append(os.path.dirname(path))
            ip_config.InteractiveShellApp.exec_lines = [STARTUP_FMT.format(session, lockfile)]
        else:
            session_code = 'from concert.quantities import q'
            ip_config.InteractiveShellApp.exec_lines = [session_code]

        ip_config.InteractiveShellApp.gui = 'asyncio'
        # This is the most robust way when taking virtualenv into account I have found so far
        ip_config.InteractiveShellApp.exec_files = [os.path.join(concert.__path__[0],
                                                                 '_ipython_setup.py')]
        ip_config.TerminalInteractiveShell.prompts_class = get_prompt_config(path or 'concert')
        ip_config.TerminalInteractiveShell.loop_runner = 'asyncio'
        ip_config.TerminalInteractiveShell.autoawait = True
        ip_config.InteractiveShell.confirm_exit = False
        ip_config.TerminalIPythonApp.display_banner = False
        ip_config.Completer.use_jedi = False

        # Now we start ipython with our configuration
        IPython.start_ipython(argv=[], config=ip_config)


class DocsCommand(SubCommand):
    """Create documentation of *session* docstring."""

    def __init__(self):
        opts = {'session': {'type': str, 'metavar': 'session'}}
        super(DocsCommand, self).__init__('docs', opts)

    def run(self, session):
        import subprocess
        import shlex

        try:
            subprocess.check_output(['pandoc', '-v'])
        except OSError:
            print("Please install pandoc and pdftex to generate docs.")
            sys.exit(1)

        cs.exit_if_not_exists(session)
        docstring = cs.get_docstring(session)

        if not docstring:
            print("No docstring in `{}' found".format(session))

        cmd_line = shlex.split('pandoc -f markdown -t latex -o {}.pdf'.format(session))
        pandoc = subprocess.Popen(cmd_line, stdin=subprocess.PIPE)
        pandoc.communicate(docstring.encode('utf-8'))


class BroadcastCommand(SubCommand):
    """Broadcast images from one source to multiple endpoints over zmq sockets either set up on the
    command line or via a yaml file.
    """

    def __init__(self):
        opts = {
            '--source': {
                'type': str,
                'help': 'Data source endpoint in form transport://address'
            },
            '--destinations': {
                'type': str,
                'nargs': '+',
                'help': "A sequence of endpoint,reliable,sndhwm string triplets "
                        "(e.g. tcp://*:5555,True,1). "
                        "Endpoint is in form transport://address, if *reliable* is True use PUSH, "
                        "otherwise PUB socket type; *sndhwm* is the high water mark (use 1 for "
                        "always getting the newest image, only applicable for non-reliable case)"
            },
            '--config': {
                'type': str,
                'help': 'Read routing information from a yaml config file (takes precedence '
                        'over --source and --destinations)'
            },
            '--start-consumers': {
                'action': 'store_true',
                'help': 'For local destination endpoints, if the destination name is a valid '
                        'concert tango server, start it (applicable only when --config is specified)'
            },
            '--logfile': {
                'type': str
            },
            '--loglevel': {
                'choices': ['perfdebug', 'aiodebug', 'debug', 'info',
                            'warning', 'error', 'critical'],
                'default': 'info'
            },
        }
        super().__init__('broadcast', opts)

    def run(self, source=None, destinations=None, config=None, start_consumers=False,
            logfile=None, loglevel=None):
        from concert.networking.base import ZmqBroadcaster

        setup_logging('broadcast', to_stream=True, filename=logfile, loglevel=loglevel)

        endpoints = []
        consumer_commands = []
        consumers = {}
        if config:
            import yaml
            from concert.ext.cmd.tango import SERVER_NAMES
            from concert.networking.base import is_zmq_endpoint_local

            with open(config) as f:
                cfg = yaml.load(f, yaml.SafeLoader)

            source = cfg['server']
            for dest, values in cfg['destinations'].items():
                endpoints.append((values['address'], values['reliable'], values.get('sndhwm', 0)))
                if endpoints[-1][-1] < 0:
                    # sndhwm check
                    raise ValueError('sndhwm must be a positive integer')
                if (
                        start_consumers and dest in SERVER_NAMES
                        and is_zmq_endpoint_local(values['address'])
                ):
                    cmd = f'concert tango {dest} --loglevel {loglevel}'
                    if 'tango-port' in values:
                        cmd += f' --port {values["tango-port"]}'
                    if logfile:
                        # Put the logs to the directory of *logfile* and use modified endpoint names
                        # for identification
                        dest_logfile = os.path.join(
                            os.path.dirname(logfile),
                            f"{dest}{values['address'].split('//')[1].replace('/', '_')}.log"
                        )
                        cmd += f' --logfile {dest_logfile}'
                    consumer_commands.append(cmd)
        else:
            for destination in destinations:
                endpoint, reliable, sndhwm = destination.split(',')
                if reliable.lower() == 'true':
                    reliable = True
                elif reliable.lower() == 'false':
                    reliable = False
                else:
                    print("Reliable must be one of `True' or `False'")
                    return

                try:
                    sndhwm = int(sndhwm)
                    if sndhwm < 0:
                        raise ValueError('sndhwm must be a positive integer')
                except Exception:
                    print('sndhwm must be a positive integer')
                    return

                endpoints.append((endpoint, reliable, sndhwm))

        LOG.info("Listening on `%s' and forwarding to `%s'", source, endpoints)
        server = ZmqBroadcaster(source, endpoints)

        if start_consumers:
            for cmd in consumer_commands:
                LOG.info("Starting consumer `%s'", cmd)
                # Consider using asyncio.create_subprocess_exec in case we need to wait for
                # processes or communicate with them
                consumers[cmd] = subprocess.Popen(cmd.split())

        async def amain():
            try:
                await server.serve()
            except BaseException as e:
                await server.shutdown()
                raise

        try:
            asyncio.run(amain())
        except KeyboardInterrupt:
            for cmd, proc in consumers.items():
                if not proc.returncode:
                    LOG.info("Terminating consumer `%s'", cmd)
                    proc.terminate()


class SessionLockException(Exception):
    """Raised if a locked session is tried to be started."""

    def __init__(self, session, lockfile):
        self.msg = (f"Tried to start session {session} with present lockfile. "
                    f"If you are sure there are no running instances, delete `{lockfile}'and start again.")

    def __str__(self):
        return self.msg


class DeviceLockException(Exception):
    """Raised if a session with present device locks is tried to be started"""

    def __init__(self, devices, lockfiles):
        self.msg = (f"Tried to start session with present device locks {devices}. "
                    f"If you are sure there are no using these devices, delete `{lockfiles}'and start again.")

    def __str__(self):
        return self.msg
