import numpy as np
from concert.devices.motors.dummy import ContinuousRotationMotor
from concert.quantities import q
from concert.imageprocessing import (compute_rotation_axis, normalize, find_sphere_centers)
from concert.measures import rotation_axis
from concert.tests import suppressed_logging, slow, assert_almost_equal, TestCase
from concert.tests.util.rotationaxis import SimulationCamera


@slow
@suppressed_logging
def test_rotation_axis():
    """
    Test tomographic rotation axis finding. The sample is a triangle which
    is offcentered by different values.
    """
    def triangle(n, width, position, left=True):
        tr = np.zeros((width, width), dtype=np.float)
        extended = np.zeros((n, n), dtype=np.float)

        indices = np.tril_indices(width)
        tr[indices] = 1
        if not left:
            tr = tr[:, ::-1]
        extended[n / 2 - width / 2: n / 2 + width / 2, position:position + width] = tr

        return extended

    def test_axis(n, width, left_position, right_position):
        left = triangle(n, width, left_position)
        right = triangle(n, width, right_position, left=False)

        center = compute_rotation_axis(left, right)
        truth = (left_position + right_position + width) / 2.0 * q.px
        assert_almost_equal(center, truth)

    n = 128
    width = 32

    # Axis is exactly in the middle
    test_axis(n, width, n / 2 - width, n / 2)

    # Axis is to the left from the center
    test_axis(n, width, 8, n / 2 + width / 4)

    # Axis is to the right from the center
    test_axis(n, width, 8, n - width)


@suppressed_logging
def test_rescale():
    array = np.array([-2.5, -1, 14, 100])

    def run_test(minimum, maximum):
        conversion = float(array.max() - array.min()) / (maximum - minimum)
        normed = normalize(array, minimum=minimum, maximum=maximum)

        np.testing.assert_almost_equal(np.gradient(array), np.gradient(normed) * conversion)
        np.testing.assert_almost_equal(normed[0], minimum)
        np.testing.assert_almost_equal(normed[-1], maximum)

    run_test(0, 1)
    run_test(-10, 47.5)


@slow
class TestSphereSegmentation(TestCase):

    def setUp(self):
        super(TestSphereSegmentation, self).setUp()
        self.x_motor = ContinuousRotationMotor()
        self.y_motor = ContinuousRotationMotor()
        self.z_motor = ContinuousRotationMotor()

        self.x_motor.position = 13 * q.deg
        self.y_motor.position = 0 * q.deg
        self.z_motor.position = -7 * q.deg

        self.camera = SimulationCamera(128, self.x_motor["position"],
                                       self.y_motor["position"],
                                       self.z_motor["position"],
                                       scales=(0.5, 0.5, 0.5),
                                       y_position=0)

    def acquire_frames(self, num_frames=10):
        da = 360 * q.degree / num_frames
        images = []
        centers = []

        for i in range(10):
            self.y_motor.move(da).join()
            images.append(self.camera.grab())
            centers.append(self.camera.ellipsoid_center)

        return (images, centers)

    def check(self, gt, measured):
        assert np.all(np.abs(np.fliplr(gt) - measured) < 1)

    def test_sphere_fully_inside(self):
        (frames, gt) = self.acquire_frames()
        centers = find_sphere_centers(frames)
        self.check(gt, centers)

    def test_sphere_some_partially_outside(self):
        self.camera.rotation_radius = self.camera.size / 2
        (frames, gt) = self.acquire_frames()
        centers = find_sphere_centers(frames)
        self.check(gt, centers)

    def test_sphere_all_partially_outside(self):
        self.camera.size = 128
        self.camera.rotation_radius = self.camera.size / 2
        self.camera.scale = (.25, .25, .25)
        frames = self.acquire_frames()[0]
        centers = find_sphere_centers(frames, correlation_threshold=0.9)
        roll, pitch = rotation_axis(centers)[:2]

        # No sphere is completely in the FOV, filter the ones with low correlation coefficient and
        # at least coarsely close match should be found to the ellipse
        assert np.abs(roll - self.z_motor.position) < 1 * q.deg
        assert np.abs(pitch - self.x_motor.position) < 1 * q.deg
