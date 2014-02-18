import os
import nose.plugins
import concert.config


class DisableAsync(nose.plugins.Plugin):
    name = 'disable_async'

    def options(self, parser, env=os.environ):
        parser.add_option('--disable-async', action='store_true',
                          default=False, dest='disable_async',
                          help="Disable asynchronous execution.")
        super(DisableAsync, self).options(parser, env=env)

    def configure(self, options, conf):
        if options.disable_async:
            concert.config.DISABLE_ASYNC = True

        super(DisableAsync, self).configure(options, conf)


class DisableGevent(nose.plugins.Plugin):
    name = 'disable_gevent'

    def options(self, parser, env=os.environ):
        parser.add_option('--disable-gevent', action='store_true',
                          default=False, dest='disable_gevent',
                          help="Disable Gevent.")
        super(DisableGevent, self).options(parser, env=env)

    def configure(self, options, conf):
        concert.config.DISABLE_GEVENT = options.disable_gevent
        super(DisableGevent, self).configure(options, conf)
