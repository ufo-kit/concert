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
        from concert.ext.tangoservers import benchmarking, camera, reco, writer

        setup_logging(server, to_stream=True, filename=logfile, loglevel=loglevel)

        SERVERS = {
            'benchmarker': {'class': benchmarking.TangoBenchmarker, 'id': 1235},
            'dummycamera': {'class': camera.TangoDummyCamera, 'id': 1236},
            'filecamera': {'class': camera.TangoFileCamera, 'id': 1236},
            'reco': {'class': reco.TangoOnlineReconstruction, 'id': 1237},
            'writer': {'class': writer.TangoWriter, 'id': 1238},
        }

        SERVERS[server]['class'].run_server(
            args=[
                'name',
                '-ORBendPoint',
                'giop:tcp::{}'.format(SERVERS[server]['id']),
                '-v4',
                '-nodb',
                '-dlist',
                'a/b/c'
            ],
            green_mode=tango.GreenMode.Asyncio
        )
