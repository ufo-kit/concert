"""
Test imaging experiments (synchrotron and X-Ray tube based).
"""
import shutil
import tempfile
import numpy as np
from concert.quantities import q
from concert.experiments.addons import ImageWriter, Accumulator
from concert.devices.cameras.dummy import Camera
from concert.devices.motors.dummy import LinearMotor, RotationMotor, ContinuousRotationMotor, \
    ContinuousLinearMotor
from concert.devices.xraytubes.dummy import XRayTube
from concert.devices.shutters.dummy import Shutter
from concert.tests import TestCase, slow
from concert.storage import DirectoryWalker

from concert.experiments.synchrotron import Radiography as SynchrotronRadiography, \
    SteppedTomography as SynchrotronSteppedTomography, \
    ContinuousTomography as SynchrotronContinuousTomography, \
    SteppedSpiralTomography as SynchrotronSteppedSpiralTomography, \
    ContinuousSpiralTomography as SynchrotronContinuousSpiralTomography

from concert.experiments.xraytube import Radiography as XRayTubeRadiography, \
    SteppedTomography as XRayTubeSteppedTomography, \
    ContinuousTomography as XRayTubeContinuousTomography, \
    SteppedSpiralTomography as XRayTubeSteppedSpiralTomography, \
    ContinuousSpiralTomography as XRayTubeContinuousSpiralTomography


class LoggingCamera(Camera):
    """
    Camera that stores information about the source and the relevant motors in its frames for
    testing.
    """
    def __init__(self, tomo_axis=None, flat_axis=None, vertical_axis=None, source=None):
        self.tomo_axis = tomo_axis
        self.flat_axis = flat_axis
        self.vertical_axis = vertical_axis
        self.source = source

        self._last_tomo_axis_position = None
        self._last_tomo_axis_velocity = None

        self._last_flat_axis_position = None
        self._last_flat_axis_velocity = None

        self._last_vertical_axis_position = None
        self._last_vertical_axis_velocity = None

        self._last_source_state = None

        Camera.__init__(self)
        self.exposure_time = 0.001 * q.s

    async def _trigger_real(self):
        if self.tomo_axis is not None:
            self._last_tomo_axis_position = await self.tomo_axis.get_position()
            try:
                if await self.tomo_axis.get_state() == "moving":
                    self._last_tomo_axis_velocity = await self.tomo_axis.get_motion_velocity()
                else:
                    self._last_tomo_axis_velocity = 0 * q.deg / q.s
            except:
                self._last_tomo_axis_velocity = 0 * q.deg / q.s

        if self.flat_axis is not None:
            self._last_flat_axis_position = await self.flat_axis.get_position()
            try:
                self._last_flat_axis_velocity = await self.flat_axis.get_velocity()
            except:
                self._last_flat_axis_velocity = 0 * q.mm / q.s

        if self.vertical_axis is not None:
            self._last_vertical_axis_position = await self.vertical_axis.get_position()
            try:
                if await self.vertical_axis.get_state() == "moving":
                    self._last_vertical_axis_velocity = await self.vertical_axis.get_motion_velocity()
                else:
                    self._last_vertical_axis_velocity = 0 * q.mm / q.s
            except:
                self._last_vertical_axis_velocity = 0 * q.mm / q.s

        if self.source is not None:
            s = await self.source.get_state()
            if s in ["on", "open"]:
                self._last_source_state = 1.0
            else:
                self._last_source_state = 0.0

    async def _grab_real(self):
        """
        Yields frames containing information about the motors and source

        0,0: Position of the tomo motor in deg
        1,0: Velocity of the tomo motor in deg/s
        0,1: Position of the flatfield motor in mm
        1,1: Velocity of the flatfield motor in mm/s
        0,2: Position of the vertical motor in mm
        1,2: Velocity of the vertical motor in mm/s
        0,3: State of the source. 1 = sample is exposed, 0 = beam is off.
        """
        if await self.get_trigger_source() == "AUTO":
            await self.trigger()
        frame = np.zeros([2, 4], dtype=np.float32)
        if self.tomo_axis is not None:
            frame[0, 0] = self._last_tomo_axis_position.to(q.deg).magnitude
            frame[1, 0] = self._last_tomo_axis_velocity.to(q.deg/q.s).magnitude

        if self.flat_axis is not None:
            frame[0, 1] = self._last_flat_axis_position.to(q.mm).magnitude
            frame[1, 1] = self._last_flat_axis_velocity.to(q.mm/q.s).magnitude

        if self.vertical_axis is not None:
            frame[0, 2] = self._last_vertical_axis_position.to(q.mm).magnitude
            frame[1, 2] = self._last_vertical_axis_velocity.to(q.mm/q.s).magnitude

        if self.source is not None:
            frame[0, 3] = self._last_source_state
        return frame


@slow
class Radiography:
    """ Abstract test class for testing radiography"""
    def setUp(self):
        self.source = None
        self.exp = None
        self.flatfield_axis = LinearMotor()
        self.flatfield_axis.motion_velocity = 10000 * q.mm / q.s
        self.camera = LoggingCamera(flat_axis=self.flatfield_axis)
        self._data_dir = tempfile.mkdtemp()
        self.walker = DirectoryWalker(root=self._data_dir)

    def tearDown(self):
        shutil.rmtree(self._data_dir)

    async def asyncSetUp(self) -> None:
        self.camera.source = self.source
        self.acc = Accumulator(self.exp.acquisitions)
        self.writer = ImageWriter(walker=self.walker, acquisitions=self.exp.acquisitions)
        await self.exp.run()

    async def test_flats(self):
        """
        Test flat acquisition

        This tests:
        - Correct number of frames recorded
        - Correct position of flatfield motor for all frames
        - Correct exposure for all frames
        """
        self.assertEqual(len(self.acc.items[self.exp.get_acquisition("flats")]),
                         await self.exp.get_num_flats())
        for flat in self.acc.items[self.exp.get_acquisition("flats")]:
            self.assertEqual(flat[0, 1], (await self.exp.get_flat_position()).to(q.mm).magnitude)
            self.assertEqual(flat[0, 3], 1.0)

    async def test_darks(self):
        """
        Test dark acquisition

        This tests:
        - Correct number of frames recorded
        - Correct exposure for all frames
        """
        self.assertEqual(len(self.acc.items[self.exp.get_acquisition("darks")]),
                         await self.exp.get_num_darks())
        for dark in self.acc.items[self.exp.get_acquisition("darks")]:
            self.assertEqual(dark[0, 3], 0.0)

    async def test_radios(self):
        """
        Test radio acquisition

        This tests:
        - Correct number of frames recorded
        - Correct position of flatfield motor for all frames
        - Correct exposure for all frames
        """
        self.assertEqual(len(self.acc.items[self.exp.get_acquisition("radios")]),
                         await self.exp.get_num_projections())
        for flat in self.acc.items[self.exp.get_acquisition("radios")]:
            self.assertEqual(flat[0, 1], (await self.exp.get_radio_position()).to(q.mm).magnitude)
            self.assertEqual(flat[0, 3], 1.0)

    async def test_finish_states(self):
        self.assertEqual(await self.camera.get_state(), "standby")
        source_state = await self.source.get_state() in ["off", "closed"]
        self.assertTrue(source_state, msg="Source state test")


@slow
class SteppedTomography(Radiography):
    """ Abstract test class for testing stepped tomography"""
    def setUp(self):
        Radiography.setUp(self)
        self.tomo_motor = RotationMotor()
        self.tomo_motor.motion_velocity = 20000 * q.deg / q.s
        self.camera.tomo_axis = self.tomo_motor

    async def test_radios(self):
        """
        Test projection acquisition for stepped tomography

        This tests:
        - Position of flat motor for all frames (in TestRadiography.test_radios())
        - source state for all frames (in TestRadiography.test_radios())
        - correct angular position for all frames
        """
        await Radiography.test_radios(self)

        steps_per_tomogram = await self.exp.get_num_projections()
        for i in range(len(self.acc.items[self.exp.get_acquisition("radios")])):
            radio = self.acc.items[self.exp.get_acquisition("radios")][i]
            tomo_position = i * (await self.exp.get_angular_range()) / steps_per_tomogram
            self.assertAlmostEqual(radio[0, 0], tomo_position.to(q.deg).magnitude, delta=1e-4)


@slow
class ContinuousTomography(Radiography):
    """ Abstract test class for testing continuous tomography"""
    def setUp(self):
        Radiography.setUp(self)
        self.tomo_motor = ContinuousRotationMotor()
        self.tomo_motor.motion_velocity = 20000 * q.deg / q.s
        self.camera.tomo_axis = self.tomo_motor

    async def test_radios(self):
        """
        Test projection acquisition for stepped tomography

        This tests:
        - Position of flat motor for all frames (in TestRadiography.test_radios())
        - source state for all frames (in TestRadiography.test_radios())
        - correct angular velocity for all frames
        """
        await Radiography.test_radios(self)

        for i in range(len(self.acc.items[self.exp.get_acquisition("radios")])):
            radio = self.acc.items[self.exp.get_acquisition("radios")][i]
            self.assertAlmostEqual(radio[1, 0],
                                   (await self.exp.get_velocity()).to(q.deg / q.s).magnitude,
                                   delta=1e-4)


@slow
class SteppedSpiralTomography(Radiography):
    """ Abstract test class for testing stepped spiral tomography"""
    def setUp(self):
        Radiography.setUp(self)
        self.tomo_motor = RotationMotor()
        self.tomo_motor.motion_velocity = 20000 * q.deg / q.s
        self.vertical_motor = LinearMotor()
        self.vertical_motor.motion_velocity = 10000 * q.mm / q.s

        self.camera.tomo_axis = self.tomo_motor
        self.camera.vertical_axis = self.vertical_motor

    async def test_radios(self):
        """
        Test projection acquisition for stepped tomography

        This tests:
        - Position of flat motor for all frames
        - source state for all frames
        - correct angular position for all frames
        """
        self.assertEqual(len(self.acc.items[self.exp.get_acquisition("radios")]),
                         await self.exp.get_num_projections() * await self.exp.get_num_tomograms())
        for radio in self.acc.items[self.exp.get_acquisition("radios")]:
            self.assertEqual(radio[0, 1], (await self.exp.get_radio_position()).to(q.mm).magnitude)
            self.assertEqual(radio[0, 3], 1.0)

        steps_per_tomogram = await self.exp.get_num_projections()
        vertical_shift_per_tomogram = await self.exp.get_vertical_shift_per_tomogram()

        for i in range(len(self.acc.items[self.exp.get_acquisition("radios")])):
            radio = self.acc.items[self.exp.get_acquisition("radios")][i]
            tomo_position = i * (await self.exp.get_angular_range()) / steps_per_tomogram
            vertical_position = i * vertical_shift_per_tomogram / steps_per_tomogram
            self.assertAlmostEqual(radio[0, 0], tomo_position.to(q.deg).magnitude, delta=1e-4)
            self.assertAlmostEqual(radio[0, 2], vertical_position.to(q.mm).magnitude, delta=1e-4)


@slow
class ContinuousSpiralTomography(Radiography):
    """ Abstract test class for testing continuous spiral tomography"""
    def setUp(self):
        Radiography.setUp(self)
        self.tomo_motor = ContinuousRotationMotor()
        self.tomo_motor.motion_velocity = 20000 * q.deg / q.s
        self.vertical_motor = ContinuousLinearMotor()
        self.vertical_motor.motion_velocity = 10000 * q.mm / q.s

        self.camera.tomo_axis = self.tomo_motor
        self.camera.vertical_axis = self.vertical_motor

    async def test_radios(self):
        """
        Test projection acquisition for stepped tomography

        This tests:
        - Position of flat motor for all frames
        - source state for all frames
        - correct angular position for all frames
        """
        self.assertEqual(len(self.acc.items[self.exp.get_acquisition("radios")]),
                         await self.exp.get_num_projections() * await self.exp.get_num_tomograms())
        time_per_tomogram = await self.exp.get_angular_range() / await self.exp.get_velocity()
        vertical_velocity = await self.exp.get_vertical_shift_per_tomogram() / time_per_tomogram

        for i in range(len(self.acc.items[self.exp.get_acquisition("radios")])):
            radio = self.acc.items[self.exp.get_acquisition("radios")][i]
            self.assertAlmostEqual(radio[1, 0],
                                   (await self.exp.get_velocity()).to(q.deg / q.s).magnitude,
                                   delta=1e-4)
            self.assertAlmostEqual(radio[1, 2], vertical_velocity.to(q.mm / q.s).magnitude,
                                   delta=1e-4)


@slow
class TestXRayTubeRadiography(Radiography, TestCase):
    """ Test implementation for XRayTubeRadiography """
    def setUp(self):
        Radiography.setUp(self)
        self.source = XRayTube()
        self.exp = XRayTubeRadiography(walker=self.walker,
                                       flat_motor=self.flatfield_axis,
                                       radio_position=0*q.mm,
                                       flat_position=10*q.mm,
                                       camera=self.camera,
                                       xray_tube=self.source,
                                       num_flats=5,
                                       num_darks=5,
                                       num_projections=10)


@slow
class TestSynchrotronRadiography(Radiography, TestCase):
    """ Test implementation for SynchrotronRadiography """
    def setUp(self):
        Radiography.setUp(self)
        self.source = Shutter()
        self.exp = SynchrotronRadiography(walker=self.walker,
                                          flat_motor=self.flatfield_axis,
                                          radio_position=0 * q.mm,
                                          flat_position=10 * q.mm,
                                          camera=self.camera,
                                          shutter=self.source,
                                          num_flats=5,
                                          num_darks=5,
                                          num_projections=10)


@slow
class TestXRayTubeSteppedTomography(SteppedTomography, TestCase):
    """ Test implementation for XRayTubeSteppedTomography """
    def setUp(self):
        SteppedTomography.setUp(self)
        self.source = XRayTube()
        self.exp = XRayTubeSteppedTomography(walker=self.walker,
                                             flat_motor=self.flatfield_axis,
                                             tomography_motor=self.tomo_motor,
                                             radio_position=0 * q.mm,
                                             flat_position=10 * q.mm,
                                             camera=self.camera,
                                             xray_tube=self.source,
                                             num_flats=5,
                                             num_darks=5,
                                             num_projections=10,
                                             angular_range=360 * q.deg,
                                             start_angle=0 * q.deg)


@slow
class TestSynchrotronSteppedTomography(SteppedTomography, TestCase):
    """ Test implementation for SynchrotronSteppedTomography """
    def setUp(self):
        SteppedTomography.setUp(self)
        self.source = Shutter()
        self.exp = SynchrotronSteppedTomography(walker=self.walker,
                                                flat_motor=self.flatfield_axis,
                                                tomography_motor=self.tomo_motor,
                                                radio_position=0 * q.mm,
                                                flat_position=10 * q.mm,
                                                camera=self.camera,
                                                shutter=self.source,
                                                num_flats=5,
                                                num_darks=5,
                                                num_projections=10,
                                                angular_range=180 * q.deg,
                                                start_angle=0 * q.deg)


@slow
class TestXRayTubeContinuousTomography(ContinuousTomography, TestCase):
    """ Test implementation for XRayTubeContinuousTomography """
    def setUp(self):
        ContinuousTomography.setUp(self)
        self.source = XRayTube()
        self.exp = XRayTubeContinuousTomography(walker=self.walker,
                                                flat_motor=self.flatfield_axis,
                                                tomography_motor=self.tomo_motor,
                                                radio_position=0 * q.mm,
                                                flat_position=10 * q.mm,
                                                camera=self.camera,
                                                xray_tube=self.source,
                                                num_flats=5,
                                                num_darks=5,
                                                num_projections=10,
                                                angular_range=360 * q.deg,
                                                start_angle=0 * q.deg)


@slow
class TestSynchrotronContinuousTomography(ContinuousTomography, TestCase):
    """ Test implementation for SynchrotronContinuousTomography """
    def setUp(self):
        ContinuousTomography.setUp(self)
        self.source = Shutter()
        self.exp = SynchrotronContinuousTomography(walker=self.walker,
                                                   flat_motor=self.flatfield_axis,
                                                   tomography_motor=self.tomo_motor,
                                                   radio_position=0 * q.mm,
                                                   flat_position=10 * q.mm,
                                                   camera=self.camera,
                                                   shutter=self.source,
                                                   num_flats=5,
                                                   num_darks=5,
                                                   num_projections=10,
                                                   angular_range=180 * q.deg,
                                                   start_angle=0 * q.deg)


@slow
class TestXRayTubeSteppedSpiralTomographyTomography(SteppedSpiralTomography, TestCase):
    """ Test implementation for XRayTubeSteppedSpiralTomography """
    def setUp(self):
        SteppedSpiralTomography.setUp(self)
        self.source = XRayTube()
        self.exp = XRayTubeSteppedSpiralTomography(walker=self.walker,
                                                   flat_motor=self.flatfield_axis,
                                                   tomography_motor=self.tomo_motor,
                                                   vertical_motor=self.vertical_motor,
                                                   radio_position=0 * q.mm,
                                                   flat_position=10 * q.mm,
                                                   camera=self.camera,
                                                   xray_tube=self.source,
                                                   start_position_vertical=0 * q.mm,
                                                   vertical_shift_per_tomogram=5 * q.mm,
                                                   sample_height=10 * q.mm,
                                                   num_flats=5,
                                                   num_darks=5,
                                                   num_projections=10,
                                                   angular_range=360 * q.deg,
                                                   start_angle=0 * q.deg)


@slow
class TestSynchrotronSteppedSpiralTomographyTomography(SteppedSpiralTomography, TestCase):
    """ Test implementation for SynchrotronSteppedSpiralTomography """
    def setUp(self):
        SteppedSpiralTomography.setUp(self)
        self.source = Shutter()
        self.exp = SynchrotronSteppedSpiralTomography(walker=self.walker,
                                                      flat_motor=self.flatfield_axis,
                                                      tomography_motor=self.tomo_motor,
                                                      vertical_motor=self.vertical_motor,
                                                      radio_position=0 * q.mm,
                                                      flat_position=10 * q.mm,
                                                      camera=self.camera,
                                                      shutter=self.source,
                                                      start_position_vertical=0 * q.mm,
                                                      vertical_shift_per_tomogram=5*q.mm,
                                                      sample_height=10 * q.mm,
                                                      num_flats=5,
                                                      num_darks=5,
                                                      num_projections=10,
                                                      angular_range=180 * q.deg,
                                                      start_angle=0 * q.deg)


@slow
class TestXRayTubeContinuousSpiralTomographyTomography(ContinuousSpiralTomography, TestCase):
    """ Test implementation for XRayTubeContinuousSpiralTomography """
    def setUp(self):
        ContinuousSpiralTomography.setUp(self)
        self.source = XRayTube()
        self.exp = XRayTubeContinuousSpiralTomography(walker=self.walker,
                                                      flat_motor=self.flatfield_axis,
                                                      tomography_motor=self.tomo_motor,
                                                      vertical_motor=self.vertical_motor,
                                                      radio_position=0 * q.mm,
                                                      flat_position=10 * q.mm,
                                                      camera=self.camera,
                                                      xray_tube=self.source,
                                                      start_position_vertical=0 * q.mm,
                                                      vertical_shift_per_tomogram=5 * q.mm,
                                                      sample_height=10 * q.mm,
                                                      num_flats=5,
                                                      num_darks=5,
                                                      num_projections=10,
                                                      angular_range=360 * q.deg,
                                                      start_angle=0 * q.deg)


@slow
class TestSynchrotronContinuousSpiralTomographyTomography(ContinuousSpiralTomography, TestCase):
    """ Test implementation for SynchrotronContinuousSpiralTomography """
    def setUp(self):
        ContinuousSpiralTomography.setUp(self)
        self.source = Shutter()
        self.exp = SynchrotronContinuousSpiralTomography(walker=self.walker,
                                                         flat_motor=self.flatfield_axis,
                                                         tomography_motor=self.tomo_motor,
                                                         vertical_motor=self.vertical_motor,
                                                         radio_position=0 * q.mm,
                                                         flat_position=10 * q.mm,
                                                         camera=self.camera,
                                                         shutter=self.source,
                                                         start_position_vertical=0 * q.mm,
                                                         vertical_shift_per_tomogram=5 * q.mm,
                                                         sample_height=10 * q.mm,
                                                         num_flats=5,
                                                         num_darks=5,
                                                         num_projections=10,
                                                         angular_range=180 * q.deg,
                                                         start_angle=0 * q.deg)