"""
test_local_experiment.py
------------------
Encapsulates local experiment test cases
"""
from concert.experiments.addons import local as local_addons
from concert.quantities import q
from concert.devices.motors.dummy import LinearMotor, ContinuousRotationMotor
from concert.devices.shutters.dummy import Shutter
from concert.storage import DirectoryWalker
from concert.experiments.synchrotron import LocalContinuousTomography
from concert.tests.integration.scenarios.test_experiment import TestExperimentBase


class TestLocalExperiment(TestExperimentBase):
    """Local experiment test case."""

    async def _setup_walker(self):
        """Set up the local directory walker."""
        self._walker = await DirectoryWalker(
            root=self._root,
            bytes_per_file=2 ** 40)

    async def _setup_experiment(self):
        """Set up the local experiment."""
        shutter = await Shutter()
        flat_motor = await LinearMotor()
        tomo = await ContinuousRotationMotor()
        self._exp = await LocalContinuousTomography(
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
        _ = await local_addons.ImageWriter(self._exp)


if __name__ == "__main__":
    pass
