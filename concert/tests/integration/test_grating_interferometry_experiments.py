import tempfile
import shutil
import numpy as np
from concert.quantities import q
from concert.tests import TestCase, slow
from concert.storage import DirectoryWalker
from concert.experiments.addons import ImageWriter, Accumulator, \
    PhaseGratingSteppingFourierProcessing
from concert.devices.cameras.dummy import Camera
from concert.devices.motors.dummy import LinearMotor
from concert.devices.xraytubes.dummy import XRayTube
from concert.devices.shutters.dummy import Shutter
from concert.experiments.synchrotron import \
    GratingInterferometryStepping as SynchrotronPhaseStepping
from concert.experiments.xraytube import GratingInterferometryStepping as XRayTubePhaseStepping
from concert.experiments.xraytube import XrayTubeMixin
from concert.experiments.synchrotron import SynchrotronMixin
from concert.experiments.imaging import Tomography, ContinuousTomography


class LoggingCamera(Camera):
    async def __ainit__(self):
        self.experiment = None
        self._last_flat_axis_position = None
        self._last_stepping_position = None
        self._last_tomo_position = None
        self._last_tomo_velocity = None
        self._last_source_state = None
        await super().__ainit__()
        await self.set_exposure_time(0.001 * q.s)

    async def _trigger_real(self):
        if isinstance(self.experiment, SynchrotronMixin):
            source = self.experiment._shutter
        elif isinstance(self.experiment, XrayTubeMixin):
            source = self.experiment._xray_tube
        else:
            raise Exception("Experiment must implement a source.")
        s = await source.get_state()
        if s in ["on", "open"]:
            self._last_source_state = 1.0
        else:
            self._last_source_state = 0.0

        self._last_flat_axis_position = await self.experiment._flat_motor.get_position()
        self._last_stepping_position = await self.experiment._stepping_motor.get_position()

        if isinstance(self.experiment, Tomography):
            self._last_tomo_position = await self.experiment._tomograpy_motor.get_position()
        if isinstance(self.experiment, ContinuousTomography):
            self._last_tomo_velocity = await self.experiment._tomography_motor.get_velocity()

    async def _grab_real(self):
        """
        Yields frames containing information about the motors and the source

        0,0: State of the source.
        0,1: Position of the flat_motor in mm.
        1,0: Position of the stepping_motor in um.
        1,1: Ideal stepping curve. dark=0.0, reference=cos(2 pi * motor_pos/period) * 1.0 + 1,
            object=cos(2 pi * motor_pos/period + pi/2) * 0.25 + 0.5 .
        0,2: Position of the tomography_motor in deg.
        1,2: Velocity of the tomography_motor in deg/s.
        """
        if await self.get_trigger_source() == "AUTO":
            await self.trigger()
        frame = np.zeros([3, 3], dtype=np.float32)
        frame[0, 0] = self._last_source_state
        frame[0, 1] = self._last_flat_axis_position.to(q.mm).magnitude
        frame[1, 0] = self._last_stepping_position.to(q.um).magnitude
        if await self.experiment.get_acquisition("darks").get_state() == "running":
            frame[1, 1] = 0.0
        if await self.experiment.get_acquisition("reference_stepping").get_state() == "running":
            pos = self._last_stepping_position.to(q.um).magnitude / (
                await self.experiment.get_grating_period()).to(q.um).magnitude
            frame[1, 1] = 1.0 * np.cos(np.pi * 2.0 * pos) + 1.0
        if await self.experiment.get_acquisition("object_stepping").get_state() == "running":
            pos = self._last_stepping_position.to(q.um).magnitude / (
                await self.experiment.get_grating_period()).to(q.um).magnitude
            frame[1, 1] = 0.25 * np.cos(np.pi * 2.0 * pos + np.pi / 2.) + 0.5
        if self._last_tomo_position is not None:
            frame[0, 2] = self._last_tomo_position.to(q.deg).magnitude
        if self._last_tomo_velocity is not None:
            frame[1, 2] = self._last_tomo_velocity.to(q.deg / q.s).magnitude
        return frame


@slow
class GratingInterferometryStepping:
    """ Abstract class for testing phase stepping."""

    async def asyncSetUp(self):
        self.source = None
        self.exp = None
        self.flatfield_axis = await LinearMotor()
        await self.flatfield_axis.set_motion_velocity(10000 * q.mm / q.s)

        self.stepping_axis = await LinearMotor()
        self.camera = await LoggingCamera()
        self._data_dir = tempfile.mkdtemp()
        self.walker = DirectoryWalker(root=self._data_dir)

    async def run_experiment(self):
        self.camera.experiment = self.exp
        self.acc = Accumulator(self.exp.acquisitions)
        self.writer = ImageWriter(walker=self.walker, acquisitions=self.exp.acquisitions)
        self.phase_stepping_addon = PhaseGratingSteppingFourierProcessing(experiment=self.exp)
        await self.exp.run()

    def tearDown(self):
        shutil.rmtree(self._data_dir)

    async def test_darks(self):
        """
        Test dark acquisition

        This tests:
        - Correct number of frames recorded
        - Correct exposure for all frames
        """
        self.assertEqual(len(self.acc.items[self.exp.get_acquisition("darks")]),
                         await self.exp.get_num_darks())
        for flat in self.acc.items[self.exp.get_acquisition("darks")]:
            self.assertEqual(flat[0, 0], 0.0)

    async def _test_stepping(self, stepping_type):
        """
        Tests the  stepping

        This tests:
        - Correct number of frames recorded
        - Correct exposure state of all frames
        - Correct flatfield_motor position
        - Correct stepping_motor position

        :param stepping_type: type of the stepping. Can be 'reference' or 'object'
        :type stepping_type: str
        """
        if stepping_type not in ["reference", "object"]:
            raise Exception("Stepping type not known.")

        self.assertEqual(len(self.acc.items[self.exp.get_acquisition(stepping_type + "_stepping")]),
                         (await self.exp.get_num_periods()
                          * await self.exp.get_num_steps_per_period()))

        stepping_start = await self.exp.get_stepping_start_position()
        step_size = await self.exp.get_grating_period() / await self.exp.get_num_steps_per_period()
        for i in range(len(self.acc.items[self.exp.get_acquisition(stepping_type + "_stepping")])):
            radio = self.acc.items[self.exp.get_acquisition(stepping_type + "_stepping")][i]
            stepping_position = i * step_size + stepping_start
            self.assertAlmostEqual(radio[1, 0], stepping_position.to(q.um).magnitude, places=3)
            self.assertEqual(radio[0, 0], 1.0)

    async def test_reference_stepping(self):
        await self._test_stepping("reference")
        for i in range(len(self.acc.items[self.exp.get_acquisition("reference_stepping")])):
            radio = self.acc.items[self.exp.get_acquisition("reference_stepping")][i]
            self.assertEqual(radio[0, 1], (await self.exp.get_flat_position()).to(q.mm).magnitude)

    async def test_object_stepping(self):
        await self._test_stepping("object")
        for i in range(len(self.acc.items[self.exp.get_acquisition("object_stepping")])):
            radio = self.acc.items[self.exp.get_acquisition("object_stepping")][i]
            self.assertEqual(radio[0, 1], (await self.exp.get_radio_position()).to(q.mm).magnitude)

    async def test_addon(self):
        self.assertAlmostEqual(self.phase_stepping_addon.object_intensity[1, 1], 0.5, places=3,
                               msg="object intensity")
        self.assertAlmostEqual(self.phase_stepping_addon.reference_intensity[1, 1], 1.0, places=3,
                               msg="reference_intensity")
        self.assertAlmostEqual(self.phase_stepping_addon.reference_visibility[1, 1], 1.0, places=3,
                               msg="reference_visibility")
        self.assertAlmostEqual(self.phase_stepping_addon.object_visibility[1, 1], 0.5, places=3,
                               msg="object_visibility")
        self.assertAlmostEqual(self.phase_stepping_addon.reference_phase[1, 1], 0, places=3,
                               msg="reference_phase")
        self.assertAlmostEqual(self.phase_stepping_addon.object_phase[1, 1], np.pi / 2., places=3,
                               msg="object_phase")


@slow
class TestSynchrotronGratingInterferometryStepping(GratingInterferometryStepping, TestCase):
    async def asyncSetUp(self):
        await GratingInterferometryStepping.asyncSetUp(self)
        self.source = await Shutter()
        self.exp = await SynchrotronPhaseStepping(walker=self.walker,
                                                  camera=self.camera,
                                                  shutter=self.source,
                                                  flat_motor=self.flatfield_axis,
                                                  stepping_motor=self.stepping_axis,
                                                  flat_position=-10 * q.cm,
                                                  radio_position=0 * q.mm,
                                                  grating_period=2.4 * q.um,
                                                  num_darks=10,
                                                  stepping_start_position=0 * q.um,
                                                  num_periods=4,
                                                  num_steps_per_period=8,
                                                  propagation_distance=20 * q.cm,
                                                  separate_scans=True)
        await self.run_experiment()


@slow
class TestXRayTubeGratingInterferometryStepping(GratingInterferometryStepping, TestCase):
    async def asyncSetUp(self):
        await GratingInterferometryStepping.asyncSetUp(self)
        self.source = await XRayTube()
        self.exp = await XRayTubePhaseStepping(walker=self.walker,
                                               camera=self.camera,
                                               xray_tube=self.source,
                                               flat_motor=self.flatfield_axis,
                                               stepping_motor=self.stepping_axis,
                                               flat_position=-10 * q.cm,
                                               radio_position=0 * q.mm,
                                               grating_period=2.4 * q.um,
                                               num_darks=10,
                                               stepping_start_position=0 * q.um,
                                               num_periods=4,
                                               num_steps_per_period=8,
                                               propagation_distance=20 * q.cm,
                                               separate_scans=True)
        await self.run_experiment()
