import subprocess
import psutil
from concert.tests import TestCase
import asyncio
from concert.networking.base import get_tango_device
test_with_tofu = False


def is_process_running(required_arguments):
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if all(arg in proc.info['cmdline'] for arg in required_arguments):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False


async def start_tango_server_standalone(server, port, device_uri):
    if server == 'walker':
        server_executable = 'TangoRemoteWalker'
    elif server == 'dummycamera':
        server_executable = 'TangoDummyCamera'
    elif server == 'reco':
        server_executable = 'TangoOnlineReconstruction'
    else:
        raise ValueError(f"Unknown server type {server}")
    if is_process_running([server_executable, str(port)]):
        return
    else:
        process = subprocess.Popen(
            [f'run_server_detached {server_executable} test -nodb --port {port} '
             f'--dlist={device_uri}'], shell=True)
        while process.poll() is None:
            await asyncio.sleep(0.1)


async def start_tango_server_concert(server, port):
    if server not in ['walker', 'dummycamera', 'reco']:
        raise ValueError(f"Unknown server type {server}")
    if is_process_running(["tango", str(port)]):
        return
    else:
        process = subprocess.Popen([f'run_server_detached concert tango {server} --port {port}'],
                                   shell=True)
        while process.poll() is None:
            await asyncio.sleep(0.1)


class TestRemoteProcessingStartup(TestCase):
    async def asyncSetUp(self):
        await asyncio.gather(start_tango_server_standalone('walker', 1200,
                                                           'concert/tango/walker'),
                             start_tango_server_concert('walker', 1201),
                             start_tango_server_standalone('dummycamera', 1202,
                                                           'concert/tango/dummycamera'),
                             start_tango_server_concert('dummycamera', 1203))
        if test_with_tofu:
            await asyncio.gather(start_tango_server_standalone('reco', 1204,
                                                               'concert/tango/reco'),
                                 start_tango_server_concert('reco', 1205))

    async def test_walker_startup(self):

        tango_dev = get_tango_device('tango://localhost:1200/concert/tango/walker#dbase=no')
        f = await tango_dev.state()
        self.assertNotEqual(f, None)

        tango_dev = get_tango_device('tango://localhost:1201/concert/tango/walker#dbase=no')
        f = await tango_dev.state()
        self.assertNotEqual(f, None)

    async def test_camera_startup(self):
        tango_dev = get_tango_device('tango://localhost:1202/concert/tango/dummycamera#dbase=no')
        f = await tango_dev.state()
        self.assertNotEqual(f, None)

        tango_dev = get_tango_device('tango://localhost:1203/concert/tango/dummycamera#dbase=no')
        f = await tango_dev.state()
        self.assertNotEqual(f, None)

    async def test_reco_startup(self):
        if test_with_tofu:
            tango_dev = get_tango_device('tango://localhost:1204/concert/tango/reco#dbase=no')
            f = await tango_dev.state()
            self.assertNotEqual(f, None)

            tango_dev = get_tango_device('tango://localhost:1205/concert/tango/reco#dbase=no')
            f = await tango_dev.state()
            self.assertNotEqual(f, None)
