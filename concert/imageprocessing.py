"""
Image processing module for manipulating image data, e.g. filtered
backprojection, flat field correction and other operations on images.
"""

import numpy as np
import logging
from scipy import ndimage
from scipy.signal import fftconvolve
from concert.quantities import q


LOG = logging.getLogger(__name__)


def flat_correct(radio, flat, dark=None):
    """
    Flat field correction of a radiograph *radio* with *flat* field.
    If *dark* field is supplied it is taken into account as well.
    """
    return radio / flat if dark is None else (radio - dark) / (flat - dark)


def ramp_filter(width):
    """Get a 1D ramp filter for filtering sinogram rows."""
    base = np.arange(-width / 2, width / 2)

    return np.fft.fftshift(np.abs(base)) * 2.0 / width


def needle_tips(images):
    """Get sample tips from images."""
    tips = []

    for image in images:
        tip = _get_ellipse_point(image)
        if tip is not None:
            tips.append(tip)

    if len(tips) == 0:
        raise ValueError("No sample tip points found.")

    return tips


def _segment(image):
    """
    Segment a flat corrected *image* into a sample and a background.

    Assume normally distributed noise, take the full width at
    1/1000 of the maximum and make it a threshold for finding
    the sample. The sample must be highly absorbing. The baseline
    for the background is taken from the top row of the image.
    """
    thr = 0.5
    image[image < thr] = 0
    image[image >= thr] = 1
    image = 1 - image
    image = ndimage.binary_fill_holes(np.cast[np.bool](image))

    # Close the image to get rid of noise-caused fuzzy segmentation.
    return ndimage.binary_closing(image) + image


def _get_boundary_coordinates(coordinates, max_val):
    """Return coordinates which reside on image edges."""
    return [coor for coor in coordinates if coor % max_val == 0]


def _is_corner_point(point, shape):
    """Test if the *point* lies in one of the image corners."""
    return (point[1] == 0 or point[1] == shape[1] - 1) and\
        (point[0] == 0 or point[0] == shape[0] - 1)


def _get_intersection_points(image):
    """Get *image* edges and sample intersection points. The *image* is
    a segmented binary image."""
    y_ind, x_ind = np.where(image != 0)
    x_low = x_ind[np.where(y_ind == 0)]
    x_high = x_ind[np.where(y_ind == image.shape[0] - 1)]
    y_low = y_ind[np.where(x_ind == 0)]
    y_high = y_ind[np.where(x_ind == image.shape[1] - 1)]

    points = []
    if len(x_low) != 0:
        points.append((0, x_low[0]))
        if x_low[-1] != x_low[0]:
            points.append((0, x_low[-1]))
    if len(x_high) != 0:
        points.append((image.shape[0] - 1, x_high[0]))
        if x_high[-1] != x_high[0]:
            points.append((image.shape[0] - 1, x_high[-1]))
    if len(y_low) != 0:
        points.append((y_low[0], 0))
        if y_low[-1] != y_low[0]:
            points.append((y_low[-1], 0))
    if len(y_high) != 0:
        points.append((y_high[0], image.shape[1] - 1))
        if y_high[-1] != y_high[0]:
            points.append((y_high[-1], image.shape[1] - 1))

    if len(points) > 2:
        # The sample is big and besides intersection points it fills some
        # corners of the image.
        res = []
        for point in points:
            if not _is_corner_point(point, image.shape):
                res.append(point)
        points = res

    return points


def _get_axis_intersection(p_1, p_2, shape):
    """Get intersections of a vector perpendicular to a vector defined by
    *p_1* and *p_2* and image edges defined by image *shape*."""
    # First check if the center lies on an edge
    if p_1[0] == p_2[0]:
        return [(p_1[0], (p_1[1] + p_2[1]) / 2)]
    elif p_1[1] == p_2[1]:
        return [((p_1[0] + p_2[0]) / 2.0, p_1[1])]

    p_x = (p_1[1] + p_2[1]) / 2.0
    p_y = (p_1[0] + p_2[0]) / 2.0
    v_y = p_1[0] - p_2[0]
    v_x = p_2[1] - p_1[1]
    height, width = shape[0] - 1, shape[1] - 1

    left = p_y - v_x * p_x / v_y, 0
    right = p_y + v_x * (width - p_x) / v_y, width
    bottom = 0, p_x - v_y * p_y / v_x
    top = height, p_x + v_y * (height - p_y) / v_x

    res = set([left, right, bottom, top])
    # Filter intersections which are out of the image bounding box.
    res = [x for x in res if 0 <= x[0] <= height and 0 <= x[1] <= width]

    return res


def _get_sample_tip(image):
    """Extract sample tip from the labeled *image*."""
    # First check if the sample tip is in the FOV. If not, there are no
    # objects or the sample splits the image in half.
    if ndimage.label(image.max() - image)[1] != 1:
        # No sample found in this image. It is either completely out of the
        # FOV or it splits the image in half, thus the tip is out of the FOV.
        return None

    p_1, p_2 = _get_intersection_points(image)
    intersections = _get_axis_intersection(p_1, p_2, image.shape)

    if len(intersections) == 0:
        return None

    p_inter = None

    # Get the intersection where a sample is present.
    for intersection in intersections:
        if image[intersection] > 0:
            p_inter = intersection
            break

    if p_inter is None:
        return None

    y_ind, x_ind = np.where(image > 0)

    # Calculate distances from the intersection point.
    distances = np.sqrt((y_ind - p_inter[0]) ** 2 + (x_ind - p_inter[1]) ** 2)

    # Most distant points are candidates for the sample tip.
    indices = np.where(distances == distances.max())[0]

    tips = [(y_ind[i], x_ind[i]) for i in indices]
    tip_center = center_of_points(tips)

    # Now find the intersection of the object and line going through it
    # in the direction to the tip.
    x_direction = tip_center[1] - p_inter[1]
    if x_direction == 0:
        # The line is parallel with y_ind axis
        x_ind = np.ones(image.shape[1], dtype=np.int64) * p_inter[1]
        y_ind = np.arange(image.shape[0])
    else:
        y_direction = tip_center[0] - p_inter[0]
        x_ind = np.arange(image.shape[1])
        y_ind = np.cast[np.int](np.round(p_inter[0] + y_direction * (
                                         x_ind - p_inter[1]) / x_direction))

    # Cut values going beyond image boundaries.
    below = np.where(y_ind < image.shape[0])[0]
    y_ind = y_ind[below]
    x_ind = x_ind[below]
    above = np.where(0 <= y_ind)[0]
    y_ind = y_ind[above].astype(np.int)
    x_ind = x_ind[above].astype(np.int)

    # Drop indices at which there is no object in the image.
    nonzero = np.where(image[y_ind, x_ind] > 0)[0]
    x_ind = x_ind[nonzero]
    y_ind = y_ind[nonzero]

    distances = np.sqrt((y_ind - p_inter[0]) ** 2 + (x_ind - p_inter[1]) ** 2)
    i = distances.argmax()

    return y_ind[i], x_ind[i]


def _get_sample(image):
    """Segment the *image* to a sample and a background."""
    labels = _get_regions(image)
    labels = _get_boundary_regions(labels)
    # Cut the edges by one pixel because hole filling does not affect them.
    return _get_biggest_region(labels)[1:-1, 1:-1]


def _get_ellipse_point(image):
    """Extract ellipse points from the sample in *image*."""
    labels = _get_sample(_segment(image))

    return _get_sample_tip(labels)


def _get_biggest_region(image):
    """Get the region with the biggest area in the *image*."""
    labels, features = ndimage.label(image)
    sizes = ndimage.sum(image, labels, range(features + 1))
    max_feature = np.argmax(sizes)
    labels[labels != max_feature] = 0

    return labels


def _get_boundary_regions(labels):
    """Extract regions from *labels* which appear on image boundaries only."""
    features = np.max(labels)

    for i in range(features + 1):
        y_ind, x_ind = np.where(labels == i)
        if len(x_ind) != 0 and len(y_ind) != 0 and x_ind.min() != 0 and\
                x_ind.max() != labels.shape[1] - 1 and y_ind.min() != 0 and\
                y_ind.max() != labels.shape[0] - 1:
            labels[labels == i] = 0

    return labels


def _get_regions(image):
    """Extract regions from the binary *image* which are not too small."""
    labels, features = ndimage.label(image)
    sizes = ndimage.sum(image, labels, range(features + 1))

    # Remove small regions.
    mask = sizes < 100
    remove_indices = mask[labels]
    labels[remove_indices] = 0

    return labels


def center_of_points(points):
    """
    Find a simplified center of mass withouth point-weighing
    from a set of *points*.
    """
    y_ind, x_ind = zip(*points)

    c_y = float(np.sum(y_ind)) / len(points)
    c_x = float(np.sum(x_ind)) / len(points)

    return c_y, c_x


def center_of_mass(frame):
    """Calculates the center of mass of the whole frame wheighted by value."""

    frm_shape = np.array(frame.shape)
    total = frame.sum()
    if total == 0:
        return np.array([-1, -1])
    else:
        y = (frame.sum(1)*np.arange(frm_shape[0])).sum() / total
        x = (frame.sum(0)*np.arange(frm_shape[1])).sum() / total
        return np.array([y, x])


def compute_rotation_axis(first_projection, last_projection):
    """
    Compute the tomographic rotation axis based on cross-correlation technique.
    *first_projection* is the projection at 0 deg, *last_projection* is the projection
    at 180 deg.
    """
    width = first_projection.shape[1]
    first_projection = first_projection - first_projection.mean()
    last_projection = last_projection - last_projection.mean()

    # The rotation by 180 deg flips the image horizontally, in order
    # to do cross-correlation by convolution we must also flip it
    # vertically, so the image is transposed and we can apply convolution
    # which will act as cross-correlation
    convolved = fftconvolve(first_projection, last_projection[::-1, :], mode='same')
    center = np.unravel_index(convolved.argmax(), convolved.shape)[1]

    return (width / 2.0 + center) / 2 * q.px
