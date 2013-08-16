import time
import unittest
import logbook
from concert.quantities import q
from testfixtures import compare
from concert.tests import assert_almost_equal
from concert.devices.motors.dummy import Motor as DummyMotor
from concert.devices.cameras.dummy import Camera
from concert.processes.scan import Scanner, StepTomoScanner, ascan, dscan
from concert.tests import slow


def compare_sequences(first_sequence, second_sequence, assertion):
    for x, y in zip(first_sequence, second_sequence):
        assertion(x, y)


class TestScan(unittest.TestCase):

    def setUp(self):
        self.motor = DummyMotor()
        self.handler = logbook.TestHandler()
        self.handler.push_application()

    def tearDown(self):
        self.handler.pop_application()

    def handle_scan(self, parameters):
        self.positions.append(parameters[0].get().result())

    def test_ascan(self):
        self.positions = []

        ascan([(self.motor['position'], -2 * q.mm, 10 * q.mm)],
              n_intervals=4,
              handler=self.handle_scan)

        expected = [-2 * q.mm, 1 * q.mm, 4 * q.mm, 7 * q.mm, 10 * q.mm]
        compare_sequences(self.positions, expected, assert_almost_equal)

    def test_dscan(self):
        self.positions = []

        self.motor.position = 2 * q.mm
        dscan([(self.motor['position'], 2 * q.mm, 10 * q.mm)],
              n_intervals=4,
              handler=self.handle_scan)

        expected = [4 * q.mm, 6 * q.mm, 8 * q.mm, 10 * q.mm, 12 * q.mm]
        compare_sequences(self.positions, expected, assert_almost_equal)

    def test_process(self):
        def feedback():
            return self.motor.position

        scanner = Scanner(self.motor['position'], feedback)
        scanner.minimum = 1 * q.mm
        scanner.maximum = 10 * q.mm
        scanner.intervals = 10
        x, y = scanner.run().result()
        compare_sequences(x, y, self.assertEqual)

    @slow
    def test_tomo_scanner(self):
        class Stage(object):

            def __init__(self):
                self.angles = []
                self.angle = 0.0 * q.deg

            def move(self, value):
                self.angle += value
                self.angles.append(self.angle)

        class PrepareChecker(object):

            def __init__(self):
                self.prepared = []

            def prepare_dark_scan(self):
                self.prepared.append('dark')

            def prepare_flat_scan(self):
                self.prepared.append('flat')

            def prepare_proj_scan(self):
                self.prepared.append('proj')

        camera = Camera()
        stage = Stage()
        checker = PrepareChecker()
        scanner = StepTomoScanner(camera, stage,
                                  checker.prepare_dark_scan,
                                  checker.prepare_flat_scan,
                                  checker.prepare_proj_scan)

        scanner.angle = 0.5 * q.deg

        darks, flats, projections = scanner.run().result()
        self.assertEqual(len(projections), len(stage.angles))
        compare(checker.prepared, ['dark', 'flat', 'proj'])
