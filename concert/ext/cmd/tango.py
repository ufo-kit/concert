import tango
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

    def run(self, server, logfile=None, loglevel=None):
        server_class = None
        if server == "benchmarker":
            from concert.ext.tangoservers import benchmarking
            server_class = {'class': benchmarking.TangoBenchmarker, 'id': 1235}
        if server == "dummycamera":
            from concert.ext.tangoservers import camera
            server_class = {'class': camera.TangoDummyCamera, 'id': 1236}
        if server == "filecamera":
            from concert.ext.tangoservers import camera
            server_class = {'class': camera.TangoFileCamera, 'id': 1236}
        if server == "reco":
            from concert.ext.tangoservers import reco
            server_class = {'class': reco.TangoOnlineReconstruction, 'id': 1237}
        if server == "writer":
            from concert.ext.tangoservers import writer
            server_class = {'class': writer.TangoWriter, 'id': 1238}

        setup_logging(server, to_stream=True, filename=logfile, loglevel=loglevel)

        server_class['class'].run_server(
            args=[
                'name',
                '-ORBendPoint',
                'giop:tcp::{}'.format(server_class['id']),
                '-v4',
                '-nodb',
                '-dlist',
                'a/b/c'
            ],
            green_mode=tango.GreenMode.Asyncio
        )
