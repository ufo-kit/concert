import sys
import concert.session.management as cs
from concert.helpers import Command, Bunch


class SpyderCommand(Command):
    """Start session using Spyder."""
    def __init__(self):
        opts = {'session': {'type': str}}
        super(SpyderCommand, self).__init__('spyder', opts)

    def run(self, session):
        from spyderlib import spyder
        from spyderlib.config import CONF

        cs.exit_if_not_exists(session)
        app = spyder.initialize()

        # This should come from our command line parser, however, Spyder does
        # not provide a way to get the argument parser but only the parsed
        # arguments.
        opts = {'working_directory': cs.path(),
                'debug': False,
                'profile': False,
                'multithreaded': False,
                'light': False}

        # The python executable is set explicitly here, because Spyder tends
        # not to pick up the one used in a virtualenv. This should be set in a
        # profile in order to not mess with the user's settings.
        CONF.set('console', 'pythonexecutable', sys.executable)

        main = spyder.MainWindow(Bunch(opts))
        main.setup()
        main.show()
        main.open_file(cs.path(session))

        app.exec_()
