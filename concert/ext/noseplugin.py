import logging
import os
import nose.plugins


LOG = logging.getLogger(__name__)


class DisableAsync(nose.plugins.Plugin):
    name = 'disable_async'

    def options(self, parser, env=os.environ):
        parser.add_option('--disable-async', action='store_true',
                          default=False, dest='disable_async',
                          help="Disable asynchronous execution.")
        super(DisableAsync, self).options(parser, env=env)

    def configure(self, options, conf):
        if options.disable_async:
            import concert.helpers
            concert.helpers.DISABLE = True

        super(DisableAsync, self).configure(options, conf)
