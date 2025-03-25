from concert.session.utils import setup_logging, SubCommand

SERVER_NAMES = ['benchmarker', 'dummycamera', 'filecamera', 'reco', 'walker']


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

    def run(self, server: str, port: int, database: bool = False, device: [str, None] = None,
            instancelogfile=None, loglevel='info', logfile=None):
        """
        Run a Tango server

        :param server: String defining the server type. Can be one of 'benchmarker', 'dummycamera',
        'filecamera', 'reco', 'walker'.
        :type server: str
        :param port: Port to run the server on. If *database* is True, this will be ignored.
        :type port: int
        :param database: Run the server using a tango database.
        :type database: bool
        :param device: When a database is used, this is the device instance name.
            When no databse is used, this is the tango device server uri. If None, the device uri
            will be set to 'concert/tango/{server}'.
        :type device: str, None
        """

        import tango
        from tango.server import run
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
        if server == "walker":
            from concert.ext.tangoservers import walker
            server_class = {'class': walker.TangoRemoteWalker}

        setup_logging(server, to_stream=True, filename=logfile, loglevel=loglevel)

        if database:
            run([server_class['class']], device)
        else:
            if device is None:
                device = f'concert/tango/{server}'
            if tango.Release.version_info < (9, 4, 1):
                port_def = ["-ORBendPoint", f"giop:tcp::{port}"]
            else:
                port_def = ["-port", f"{port}"]
            run(
                [server_class['class']],
                args=[
                    server,
                    'name',
                    port_def[0],
                    port_def[1],
                    '-v4',
                    '-nodb',
                    '-dlist',
                    device
                ],
                green_mode=tango.GreenMode.Asyncio
            )
