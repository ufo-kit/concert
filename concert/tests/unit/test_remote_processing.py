import signal
import subprocess
import time

from concert.coroutines.base import start
from concert.quantities import q
from concert.tests import TestCase
from concert.ext.cmd.tango import TangoCommand
import asyncio
from concert.networking.base import get_tango_device
from concert.storage import RemoteDirectoryWalker
from multiprocessing import Process
import os
import tango

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
