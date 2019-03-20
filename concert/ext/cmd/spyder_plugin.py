import sys
import concert.session.management as cs
from concert.helpers import Command, Bunch


class SpyderCommand(Command):
    """Start session using Spyder."""
    def __init__(self):
        opts = {'session': {'type': str}}
        super(SpyderCommand, self).__init__('spyder', opts)

    def run(self, session):
        try:
            from spyderlib import spyder
            from spyderlib.config import CONF
            spyder_version = 2
        except ImportError:
            # Spyder 3
            import spyder.app.mainwindow as spyder
            from spyder.config.main import CONF
            spyder_version = 3

        cs.exit_if_not_exists(session)
        app = spyder.initialize()

        # This should come from our command line parser, however, Spyder does
        # not provide a way to get the argument parser but only the parsed
        # arguments.
        opts = {'working_directory': cs.path(),
                'debug': False,
                'profile': False,
                'multithreaded': False,
                'light': False,
                'new_instance': True}
        if spyder_version == 3:
            opts['project'] = None
            opts['window_title'] = 'Concert'

        # The python executable is set explicitly here, because Spyder tends
        # not to pick up the one used in a virtualenv. This should be set in a
        # profile in order to not mess with the user's settings.
        CONF.set('console', 'pythonexecutable', sys.executable)

        spyder.run_spyder(app, Bunch(opts), [cs.path(session)])
