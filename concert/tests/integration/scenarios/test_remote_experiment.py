"""
test_remote_experiment.py
------------------
Encapsulates remote experiment test cases
"""
import os
import zmq
from concert.experiments.addons import tango as tango_addons
from concert.quantities import q
from concert.devices.motors.dummy import LinearMotor, ContinuousRotationMotor
from concert.devices.shutters.dummy import Shutter
from concert.storage import RemoteDirectoryWalker
from concert.networking.base import get_tango_device
from concert.experiments.synchrotron import RemoteContinuousTomography
from concert.helpers import CommData
from concert.tests.integration.scenarios.test_experiment import TestExperimentBase

####################################################################################################
# Docker daemon creates DNS entries inside the specified network with the service names.
# We need to specify these domain names to communicate with services on a given exposed port.
# In the compose.yml we have specified `uca_camera` and `remote_walker` as service names for the mock
# camera and walker tango server processes running inside their respective containers.
# Hence, in the session we'd have to use these domain names.
# Moreover, we will use the environment variables, which were injected to container processes.
# These environment variables encapsulate relevant metadata from container orchestration,
# which is necessary to write the test cases.
####################################################################################################


class TestRemoteExperiment(TestExperimentBase):
    """Remote experiment test case."""

    async def asyncSetUp(self) -> None:
        # Set up remote-specific configuration
        walker_host = os.environ["REMOTE_WALKER_HOST"]
        walker_port = os.environ["REMOTE_WALKER_PORT"]
        walker_dn = os.environ["REMOTE_WALKER_DN"]
        self._walker_dev_uri = f"{walker_host}:{walker_port}/{walker_dn}#dbase=no"
        self._servers = {
            "walker": CommData(
                os.environ["UCA_NET_HOST"],
                port=os.environ["UCA_WALKER_ZMQ_PORT"],
                socket_type=zmq.PUSH
            )
        }
        await super().asyncSetUp()

    async def _setup_walker(self):
        """Set up the remote directory walker."""
        self._walker = await RemoteDirectoryWalker(
            device=get_tango_device(self._walker_dev_uri, timeout=30 * 60 * q.s),
            root=self._root,
            bytes_per_file=2 ** 40)

    async def _setup_experiment(self):
        """Set up the remote experiment."""
        shutter = await Shutter()
        flat_motor = await LinearMotor()
        tomo = await ContinuousRotationMotor()
        self._exp = await RemoteContinuousTomography(
            walker=self._walker,
            flat_motor=flat_motor,
            tomography_motor=tomo,
            radio_position=0 * q.mm,
            flat_position=10 * q.mm,
            camera=self._camera,
            shutter=shutter,
            num_flats=self._num_flats,
            num_darks=self._num_darks,
            num_projections=self._num_radios
        )
        _ = await tango_addons.ImageWriter(self._exp, self._servers["walker"],
                                           self._exp.acquisitions)


if __name__ == "__main__":
    pass
