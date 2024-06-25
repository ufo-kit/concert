import signal
import subprocess

from concert.coroutines.base import start
from concert.tests import TestCase
from concert.ext.cmd.tango import TangoCommand
import asyncio
from concert.networking.base import get_tango_device
from multiprocessing import Process
import os
import tango

test_with_tofu = False

from contextlib import asynccontextmanager

@asynccontextmanager
async def tango_run_concert(name: str, port:int):

    pro = subprocess.Popen(f"concert tango {name} --port {port}", stdout=subprocess.PIPE,
                           shell=True, preexec_fn=os.setsid)


    # TODO: this needs to go away!
    await asyncio.sleep(1)

    try:
        yield
    finally:
        os.killpg(os.getpgid(pro.pid), signal.SIGTERM)

@asynccontextmanager
async def tango_run_standalone(name: str, port:int, device_uri:str):
    if tango.Release.version_info < (9, 4, 1):
        port_def = f"-ORBendPoint giop:tcp::{port}"
    else:
        port_def = f"--port {port}"
    pro = subprocess.Popen(f"{name} test -nodb {port_def} -dlist {device_uri}", stdout=subprocess.PIPE,
                           shell=True, preexec_fn=os.setsid)

    # TODO: this needs to go away!
    await asyncio.sleep(1)

    try:
        yield
    finally:
        os.killpg(os.getpgid(pro.pid), signal.SIGTERM)


class TestRemoteProcessingStartup(TestCase):
    async def test_walker_startup(self):
        async with tango_run_concert('walker', 1233):
            tango_dev = get_tango_device('tango://localhost:1233/concert/tango/walker#dbase=no')
            f = await tango_dev.state()
            self.assertNotEqual(f, None)

        async with tango_run_standalone('TangoRemoteWalker', 1233, "concert/tango/walker"):
            tango_dev = get_tango_device('tango://localhost:1233/concert/tango/walker#dbase=no')
            f = await tango_dev.state()
            self.assertNotEqual(f, None)

    async def test_camera_startup(self):
        async with tango_run_concert('dummycamera', 1245):
            tango_dev = get_tango_device('tango://localhost:1245/concert/tango/dummycamera#dbase=no')
            f = await tango_dev.state()
            self.assertNotEqual(f, None)

        async with tango_run_standalone('TangoDummyCamera', 1245, "concert/tango/dummycamera"):
            tango_dev = get_tango_device('tango://localhost:1245/concert/tango/dummycamera#dbase=no')
            f = await tango_dev.state()
            self.assertNotEqual(f, None)

    async def test_reco_startup(self):
        if test_with_tofu:
            async with tango_run_concert('reco', 1247):
                tango_dev = get_tango_device('tango://localhost:1247/concert/tango/reco#dbase=no')
                f = await tango_dev.state()
                self.assertNotEqual(f, None)

            async with tango_run_standalone('reco', 1247, "concert/tango/reco"):
                tango_dev = get_tango_device('tango://localhost:1247/concert/tango/reco#dbase=no')
                f = await tango_dev.state()
                self.assertNotEqual(f, None)
