from concert.quantities import q
from concert.tests import assert_almost_equal
from concert.devices.base import Device, Parameter
from concert.devices.motors.dummy import Motor as DummyMotor
from concert.devices.cameras.dummy import Camera
from concert.processes.scan import Scanner, StepTomoScanner, ascan, dscan
from concert.tests import slow
from concert.tests.base import ConcertTest


def compare_sequences(first_sequence, second_sequence, assertion):
    for x, y in zip(first_sequence, second_sequence):
        assertion(x, y)


class TestScan(ConcertTest):

    def setUp(self):
        super(TestScan, self).setUp()
        self.motor = DummyMotor()

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

        scanner = Scanner(
            self.motor['position'], feedback, 1 * q.mm, 10 * q.mm)
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
        self.assertEqual(checker.prepared, ['dark', 'flat', 'proj'])

    def test_parameter_lock(self):
        class SomeDevice(Device):
            def __init__(self):
                params = [Parameter(name='foo', unit=q.m,
                                    fget=self._get_foo, fset=self._set_foo),
                          Parameter(name='bar', unit=q.m,
                                    fget=self._get_bar, fset=self._set_bar)]

                super(SomeDevice, self).__init__(params)
                self._foo = 1 * q.m
                self._bar = 1 * q.m

            def _get_foo(self):
                return self._foo

            def _set_foo(self, val):
                self._foo = val

            def _get_bar(self):
                return self._bar

            def _set_bar(self, val):
                self._bar = val

        device = SomeDevice()

        def scan_bar(foo):
            def feedback():
                return device.bar

            device.foo = foo
            scanner = Scanner(device['bar'], feedback, 1*q.m, 2*q.m, 10)
            return scanner.run().result()

        def scan_foo():
            def feedback():
                return scan_bar(device.foo)

            scanner = Scanner(device['foo'], feedback, 1*q.m, 2*q.m, 10)
            return scanner.run().result()

        x, y = scan_foo()
        self.assertEqual(x.shape[0], len(y))
