from concert.session.utils import setup_logging, SubCommand


SERVER_NAMES = ['benchmarker', 'dummycamera', 'filecamera', 'reco', 'writer']


class TangoCommand(SubCommand):
    """Start a Tango server"""

    def __init__(self):
        opts = {
            'server': {
                'choices': SERVER_NAMES,
                'help': 'Name of the tango server to run'
            },
            '--port': {
                'type': int,
                'help': 'Port to run the server on',
                'default': 1234
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
        super(TangoCommand, self).__init__('tango', opts)

    def run(self, server, port, logfile=None, loglevel=None):
        import tango
        server_class = None
        if server == "benchmarker":
            from concert.ext.tangoservers import benchmarking
            server_class = {'class': benchmarking.TangoBenchmarker}
        if server == "dummycamera":
            from concert.ext.tangoservers import camera
            server_class = {'class': camera.TangoDummyCamera}
        if server == "filecamera":
            from concert.ext.tangoservers import camera
            server_class = {'class': camera.TangoFileCamera}
        if server == "reco":
            from concert.ext.tangoservers import reco
            server_class = {'class': reco.TangoOnlineReconstruction}
        if server == "writer":
            from concert.ext.tangoservers import writer
            server_class = {'class': writer.TangoWriter}

        setup_logging(server, to_stream=True, filename=logfile, loglevel=loglevel)

        server_class['class'].run_server(
            args=[
                'name',
                '-ORBendPoint',
                f'giop:tcp::{port}',
                '-v4',
                '-nodb',
                '-dlist',
                f'concert/tango/{server}'
            ],
            green_mode=tango.GreenMode.Asyncio
        )
