import signal
import subprocess
import time
from random import randint
import socket

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

def get_unused_port(min_port=1000, max_port=65535):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = None
    while not port:
        port_to_test = randint(min_port, max_port)
        try:
            sock.bind(('localhost', port_to_test))
        except socket.error as e:
            continue
        port = port_to_test
    print('listening on port', port)
    return port


@asynccontextmanager
async def tango_run_concert(name: str, start_timeout = 10*q.s):
    port = get_unused_port()
    pro = subprocess.Popen(f"concert tango {name} --port {port}", stdout=subprocess.PIPE,
                           shell=True, preexec_fn=os.setsid)
    device_uri = f"concert/tango/{name}"
    start_time = time.time() * q.s
    device_uri = f"tango://localhost:{port}/{device_uri}#dbase=no"
    try:
        while True:
            try:
                d = tango.DeviceProxy(device_uri)
                d.state()
            except (tango.ConnectionFailed, tango.DevFailed):
                if time.time() * q.s - start_time > start_timeout:
                    raise TimeoutError(f"Device {device_uri} did not start in time")
                await asyncio.sleep(0.1)
                continue
            break
        yield device_uri
    finally:
        os.killpg(os.getpgid(pro.pid), signal.SIGTERM)


@asynccontextmanager
async def tango_run_standalone(name: str, start_timeout=60*q.s):
    port = get_unused_port()
    device_uri = f"concert/tango/{name}"
    if tango.Release.version_info < (9, 4, 1):
        port_def = f"-ORBendPoint giop:tcp::{port}"
    else:
        port_def = f"--port {port}"
    pro = subprocess.Popen(f"{name} test -nodb {port_def} -dlist {device_uri}", stdout=subprocess.PIPE,
                           shell=True,preexec_fn=os.setsid,  bufsize=0)

    start_time = time.time() * q.s
    device_uri = f"tango://localhost:{port}/{device_uri}#dbase=no"
    try:
        while True:
            try:
                d = tango.DeviceProxy(device_uri)
                d.state()
            except (tango.ConnectionFailed, tango.DevFailed):
                if time.time() * q.s - start_time > start_timeout:
                    raise TimeoutError(f"Device {device_uri} did not start in time")
                await asyncio.sleep(0.1)
                continue
            break
        yield device_uri
    finally:
            os.killpg(os.getpgid(pro.pid), signal.SIGTERM)


class TestRemoteProcessingStartup(TestCase):
    async def test_walker_startup(self):
        async with tango_run_concert('walker', 10*q.s) as tango_uri:
            tango_dev = get_tango_device(tango_uri)
            f = await tango_dev.state()
            self.assertNotEqual(f, None)

        async with tango_run_standalone('TangoRemoteWalker', 3*q.s) as tango_uri:
            tango_dev = get_tango_device(tango_uri)
            f = await tango_dev.state()
            self.assertNotEqual(f, None)

    async def test_reco_startup(self):
        if test_with_tofu:
            async with tango_run_concert('reco', 10*q.s) as tango_uri:
                tango_dev = get_tango_device(tango_uri)
                f = await tango_dev.state()
                self.assertNotEqual(f, None)

            async with tango_run_standalone('reco', 10*q.s) as tango_uri:
                tango_dev = get_tango_device(tango_uri)
                f = await tango_dev.state()
                self.assertNotEqual(f, None)
