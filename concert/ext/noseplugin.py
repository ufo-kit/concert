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
        concert.config.ENABLE_ASYNC = not options.disable_async
        super(DisableAsync, self).configure(options, conf)


class EnableGevent(nose.plugins.Plugin):
    name = 'enable_gevent'

    def options(self, parser, env=os.environ):
        parser.add_option('--enable-gevent', action='store_true',
                          default=False, dest='enable_gevent',
                          help="Enable Gevent.")
        super(EnableGevent, self).options(parser, env=env)

    def configure(self, options, conf):
        concert.config.ENABLE_GEVENT = options.enable_gevent
        super(EnableGevent, self).configure(options, conf)
