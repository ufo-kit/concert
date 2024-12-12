import asyncio
import os
import signal
import subprocess
import time
import tango
from concert.coroutines.base import start
from concert.quantities import q
from concert.tests import TestCase
from concert.ext.cmd.tango import TangoCommand
from concert.networking.base import get_tango_device
from concert.storage import RemoteDirectoryWalker




test_with_tofu = False

from contextlib import asynccontextmanager

@asynccontextmanager
async def tango_run_concert(name: str, port:int, start_timeout = 10*q.s):

    pro = subprocess.Popen(f"concert tango {name} --port {port}", stdout=subprocess.PIPE,
                           shell=True, preexec_fn=os.setsid)

    try:
        yield
    finally:
        os.killpg(os.getpgid(pro.pid), signal.SIGTERM)

@asynccontextmanager
async def tango_run_standalone(name: str, port:int, device_uri:str, start_timeout = 60*q.s):
    if tango.Release.version_info < (9, 4, 1):
        port_def = f"-ORBendPoint giop:tcp::{port}"
    else:
        port_def = f"--port {port}"
    pro = subprocess.Popen(f"{name} test -nodb {port_def} -dlist {device_uri}", stdout=subprocess.PIPE,
                           shell=True, preexec_fn=os.setsid)

    start_time = time.time() * q.s
    while True:
        try:
            stdout, stderr= pro.communicate(timeout=0.1)
            print(stdout.decode())
            print(stderr.decode())

            if "Ready to accept request" in stdout.decode():
                break
            if "Exited" in stdout.decode():
                os.killpg(os.getpgid(pro.pid), signal.SIGTERM)
                raise TimeoutError("Tango device exited unexpectedly")
        except subprocess.TimeoutExpired:
            if time.time() * q.s - start_time > start_timeout:
                os.killpg(os.getpgid(pro.pid), signal.SIGTERM)
                raise TimeoutError("Timeout while waiting for Tango device to start")
            continue
        if time.time() * q.s - start_time > start_timeout:
            os.killpg(os.getpgid(pro.pid), signal.SIGTERM)
            raise TimeoutError("Timeout while waiting for Tango device to start")

    try:
        yield
    finally:
        os.killpg(os.getpgid(pro.pid), signal.SIGTERM)


class TestRemoteProcessingStartup(TestCase):
    async def test_walker_startup(self):
        async with tango_run_standalone('TangoRemoteWalker', 1233, "concert/tango/walker", 20*q.s):
            tango_dev = get_tango_device('tango://localhost:1233/concert/tango/walker#dbase=no')
            f = await tango_dev.state()
            self.assertNotEqual(f, None)

    async def test_camera_startup(self):
        #async with tango_run_concert('dummycamera', 1245):
        #    tango_dev = get_tango_device('tango://localhost:1245/concert/tango/dummycamera#dbase=no')
        #    f = await tango_dev.state()
        #    self.assertNotEqual(f, None)

        async with tango_run_standalone('TangoDummyCamera', 1245, "concert/tango/dummycamera", 20*q.s):
            tango_dev = get_tango_device('tango://localhost:1245/concert/tango/dummycamera#dbase=no')
            f = await tango_dev.state()
            self.assertNotEqual(f, None)

    async def test_reco_startup(self):
        if test_with_tofu:
            async with tango_run_concert('reco', 1247):
                tango_dev = get_tango_device('tango://localhost:1247/concert/tango/reco#dbase=no')
                f = await tango_dev.state()
                self.assertNotEqual(f, None)

            async with tango_run_standalone('reco', 1247, "concert/tango/reco", 10*q.s):
                tango_dev = get_tango_device('tango://localhost:1247/concert/tango/reco#dbase=no')
                f = await tango_dev.state()
                self.assertNotEqual(f, None)

    async def test_rae_startup(self) -> None:
        async with tango_run_standalone('rae', 1248, "concert/tango/rae", 20*q.s):
            tango_dev = get_tango_device('tango://localhost:1248/concert/tango/rae#dbase=no')
            f = await tango_dev.state()
            self.assertNotEqual(f, None)


if __name__ == "__main__":
    pass
