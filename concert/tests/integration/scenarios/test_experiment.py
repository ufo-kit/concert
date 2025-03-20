"""
test_experiment.py
------------------
Encapsulates remote experiment test cases
"""
import asyncio
import os
import json
from pathlib import Path
from typing import List, Any, Dict
import shutil
from numpy import ndarray as ArrayLike
import skimage.io as skio
import zmq
import concert
from concert.experiments.addons import tango as tango_addons
from concert.quantities import q
from concert.devices.motors.dummy import LinearMotor, ContinuousRotationMotor
from concert.devices.shutters.dummy import Shutter
from concert.storage import RemoteDirectoryWalker
from concert.networking.base import get_tango_device
from concert.experiments.synchrotron import RemoteContinuousTomography
from concert.devices.cameras.uca import RemoteNetCamera
from concert.helpers import CommData
from concert.tests import TestCase
####################################################################################################
# Docker daemon creates a DNS entries inside the specified network with the service names. We need to 
# specify these domain names to communicate with a services on a given exposed port. In the compose.yml
# we have specified `uca_camera` and `remote_walker` as service names for the mock camera and walker
# tango server processes running inside their respective containers. Hence, in the session we'd
# have to use these domain names. Moreover, we will use the environment variables, which were injected
# to conainer process. These environment variables encapsulate relevant metadata from container
# orchestration, which are necessary to write the test cases.
####################################################################################################

class TestRemoteExperiment(TestCase):

    @staticmethod
    def list_files(startpath: str) -> None:
        for root, dirs, files in os.walk(startpath):
            level: str = root.replace(startpath, '').count(os.sep)
            indent: str = ' ' * 4 * (level)
            print('{}{}/'.format(indent, os.path.basename(root)))
            subindent: str = ' ' * 4 * (level + 1)
            for f in files:
                print('{}{}'.format(subindent, f))

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        walker_host = os.environ["REMOTE_WALKER_HOST"]
        walker_port = os.environ["REMOTE_WALKER_PORT"]
        walker_dn = os.environ["REMOTE_WALKER_DN"]
        self._walker_dev_uri = f"{walker_host}:{walker_port}/{walker_dn}#dbase=no"
        self._num_darks = 10
        self._num_flats = 10
        self._num_radios = 100
        self._servers = { "walker": CommData(os.environ["UCA_NET_HOST"],
                                             port=os.environ["UCA_WALKER_ZMQ_PORT"],
                                             socket_type=zmq.PUSH)}
        self._camera = await RemoteNetCamera()
        if await self._camera.get_state() == 'recording':
            await self._camera.stop_recording()
        self._root = os.environ["DATA_ROOT"]
        self._walker = await RemoteDirectoryWalker(
            device=get_tango_device(self._walker_dev_uri, timeout=30 * 60 * q.s),
            root=self._root,
            bytes_per_file=2**40)
        shutter = await Shutter()
        flat_motor = await LinearMotor()
        tomo = await ContinuousRotationMotor()
        self._exp = await RemoteContinuousTomography(walker=self._walker,
                                       flat_motor=flat_motor, 
                                       tomography_motor=tomo,
                                       radio_position=0*q.mm,
                                       flat_position=10*q.mm,
                                       camera=self._camera,
                                       shutter=shutter,
                                       num_flats=self._num_flats,
                                       num_darks=self._num_darks,
                                       num_projections=self._num_radios)
        _ = await tango_addons.ImageWriter(self._exp, self._servers["walker"],
                                           self._exp.acquisitions)
        # Run some cleanups in the mounted location before running the
        # experiment.
        base_path: Path = Path(self._root)
        items: List[str] = os.listdir(base_path)
        items = list(filter(lambda name: "scan" in name, items))
        if len(items) > 0:
            for item in items:
                abs_path: Path = base_path.joinpath(item)
                if os.path.isfile(abs_path):
                    os.remove(abs_path)
                else:
                    shutil.rmtree(abs_path)
    
    async def test_run(self) -> None:
        _ = await self._exp.run()
        base_path: Path = Path(self._root)
        items: List[str] = os.listdir(base_path)
        items = list(filter(lambda name: "scan" in name, items))
        self.assertTrue(len(items) > 0)
        self.list_files(base_path.__str__())
        for item in items:
            abs_path: Path = base_path.joinpath(item)
            self.assertTrue(os.path.exists(abs_path.joinpath("darks")))
            self.assertTrue(os.path.exists(abs_path.joinpath("flats")))
            self.assertTrue(os.path.exists(abs_path.joinpath("radios")))
            self.assertTrue(os.path.exists(abs_path.joinpath("experiment.log")))
            self.assertTrue(os.path.exists(abs_path.joinpath("experiment_start.json")))
            self.assertTrue(os.path.exists(abs_path.joinpath("experiment_finish.json")))
            darks: ArrayLike = skio.ImageCollection(abs_path.joinpath("darks/frame_000000.tif"
                                                                    ).__str__())
            flats: ArrayLike = skio.ImageCollection(abs_path.joinpath("flats/frame_000000.tif"
                                                                    ).__str__())
            radios: ArrayLike = skio.ImageCollection(abs_path.joinpath("radios/frame_000000.tif"
                                                                    ).__str__())
            print(f"Num Darks: {len(darks)}")
            print(f"Num flats: {len(flats)}")
            print(f"Num radios: {len(radios)}")
            with open(abs_path.joinpath("experiment_finish.json")) as log:
                exp_log: Dict[str, Any] = json.load(log)["experiment"]
                self.assertTrue(len(darks) == int(exp_log["num_darks"]))
                self.assertTrue(len(flats) == int(exp_log["num_flats"]))
                self.assertTrue(len(radios) == int(exp_log["num_projections"]))

    async def asyncTearDown(self) -> None:
        await super().asyncTearDown()
        base_path: Path = Path(self._root)
        items: List[str] = os.listdir(base_path)
        items = list(filter(lambda name: "scan" in name, items))
        if len(items) > 0:
            for item in items:
                abs_path: Path = base_path.joinpath(item)
                if os.path.isfile(abs_path):
                    os.remove(abs_path)
                else:
                    shutil.rmtree(abs_path)
        await self._camera.unregister_all()


if __name__ == "__main__":
    pass
