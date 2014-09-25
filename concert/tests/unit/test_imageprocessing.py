import numpy as np
from concert.quantities import q
from concert.imageprocessing import compute_rotation_axis, normalize
from concert.tests import suppressed_logging, slow, assert_almost_equal


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
