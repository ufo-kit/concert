import numpy as np
from concert.coroutines.base import async_generate
from concert.devices.motors.dummy import ContinuousRotationMotor
from concert.quantities import q
from concert.imageprocessing import (
    compute_rotation_axis,
    convert_image_to_nbit,
    find_sphere_centers,
    normalize,
)
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
        tr = np.zeros((width, width), dtype=float)
        extended = np.zeros((n, n), dtype=float)

        indices = np.tril_indices(width)
        tr[indices] = 1
        if not left:
            tr = tr[:, ::-1]
        extended[n // 2 - width // 2: n // 2 + width // 2, position:position + width] = tr

        return extended

    def test_axis(n, width, left_position, right_position):
        left = triangle(n, width, left_position)
        right = triangle(n, width, right_position, left=False)

        center = compute_rotation_axis(left, right)
        truth = (left_position + right_position + width) / 2 * q.px
        assert_almost_equal(center, truth)

    n = 128
    width = 32

    # Axis is exactly in the middle
    test_axis(n, width, n // 2 - width, n // 2)

    # Axis is to the left from the center
    test_axis(n, width, 8, n // 2 + width // 4)

    # Axis is to the right from the center
    test_axis(n, width, 8, n - width)


@suppressed_logging
def test_rescale():
    array = np.array([-2.5, -1, 14, 100])

    def run_test(minimum, maximum):
        conversion = (array.max() - array.min()) / (maximum - minimum)
        normed = normalize(array, minimum=minimum, maximum=maximum)

        np.testing.assert_almost_equal(np.gradient(array), np.gradient(normed) * conversion)
        np.testing.assert_almost_equal(normed[0], minimum)
        np.testing.assert_almost_equal(normed[-1], maximum)

    run_test(0, 1)
    run_test(-10, 47.5)


def test_convert_image_to_nbit():
    image = np.arange(100, dtype=np.float32).reshape(10, 10)

    for num_bits, dtype in (
        (1, np.uint8),
        (8, np.uint8),
        (9, np.uint16),
        (16, np.uint16),
        (17, np.uint32),
        (32, np.uint32),
        (33, np.uint64),
        (64, np.uint64),
    ):
        converted = convert_image_to_nbit(image, num_bits=num_bits, percentile=0)
        assert converted.dtype == dtype
        assert converted.shape == image.shape
        assert converted.min() == 0
        assert converted.max() == 2 ** num_bits - 1


def test_convert_image_to_nbit_percentile_and_channels():
    image = np.array([[0, 1], [2, 100]], dtype=np.float32)
    converted = convert_image_to_nbit(image, num_bits=8, percentile=25, num_channels=3)

    assert converted.shape == (2, 2, 3)
    assert converted.dtype == np.uint8
    np.testing.assert_array_equal(converted[..., 0], converted[..., 1])
    np.testing.assert_array_equal(converted[..., 1], converted[..., 2])
    assert converted[0, 0, 0] == 0
    assert converted[-1, -1, 0] == 255


def test_convert_image_to_nbit_constant_and_invalid_arguments():
    converted = convert_image_to_nbit(np.ones((2, 3)), num_bits=12, num_channels=2)
    assert converted.shape == (2, 3, 2)
    assert converted.dtype == np.uint16
    assert not converted.any()

    for num_bits in (0, 65):
        with np.testing.assert_raises(ValueError):
            convert_image_to_nbit(np.ones((2, 2)), num_bits=num_bits)
    with np.testing.assert_raises(ValueError):
        convert_image_to_nbit(np.ones((2, 2)), num_channels=0)


@slow
class TestSphereSegmentation(TestCase):

    async def asyncSetUp(self):
        await super(TestSphereSegmentation, self).asyncSetUp()
        self.x_motor = await ContinuousRotationMotor()
        self.y_motor = await ContinuousRotationMotor()
        self.z_motor = await ContinuousRotationMotor()

        await self.x_motor.set_position(13 * q.deg)
        await self.y_motor.set_position(0 * q.deg)
        await self.z_motor.set_position(-7 * q.deg)

        self.camera = await SimulationCamera(128, self.x_motor["position"],
                                             self.y_motor["position"],
                                             self.z_motor["position"],
                                             scales=(0.5, 0.5, 0.5),
                                             y_position=0)

    async def acquire_frames(self, num_frames=10):
        da = 360 * q.degree / num_frames
        images = []
        centers = []

        async with self.camera.recording():
            for i in range(10):
                await self.y_motor.move(da)
                images.append(await self.camera.grab())
                centers.append(self.camera.ellipsoid_center)

        return (images, centers)

    def check(self, gt, measured):
        assert np.all(np.abs(np.fliplr(gt) - measured) < 1)

    async def test_sphere_fully_inside(self):
        (frames, gt) = await self.acquire_frames()
        centers = await find_sphere_centers(async_generate(frames))
        self.check(gt, centers)

    async def test_sphere_some_partially_outside(self):
        self.camera.rotation_radius = self.camera.size // 2
        (frames, gt) = await self.acquire_frames()
        centers = await find_sphere_centers(async_generate(frames))
        self.check(gt, centers)

    async def test_sphere_all_partially_outside(self):
        self.camera.size = 128
        self.camera.rotation_radius = self.camera.size // 2
        self.camera.scale = (.25, .25, .25)
        frames = (await self.acquire_frames())[0]
        centers = await find_sphere_centers(async_generate(frames), correlation_threshold=0.9)
        roll, pitch = rotation_axis(centers)[:2]

        # No sphere is completely in the FOV, filter the ones with low correlation coefficient and
        # at least coarsely close match should be found to the ellipse
        assert np.abs(roll - await self.z_motor.get_position()) < 1 * q.deg
        assert np.abs(pitch - await self.x_motor.get_position()) < 1 * q.deg
