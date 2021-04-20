import os
import nose.plugins
import concert.config


class DisableAsync(nose.plugins.Plugin):
    name = 'disable_casync'

    def options(self, parser, env=os.environ):
        parser.add_option('--disable-casync', action='store_true',
                          default=False, dest='disable_casync',
                          help="Disable casynchronous execution.")
        super(DisableAsync, self).options(parser, env=env)

    def configure(self, options, conf):
        concert.config.ENABLE_ASYNC = not options.disable_casync
        concert.config.PRINT_NOASYNC_EXCEPTION = False
        super(DisableAsync, self).configure(options, conf)
