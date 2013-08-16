"""
The module determines a 3D circle normal inclination angles towards the
:math:`y`-axis (0,1,0) and the center of the circle.

**Algorithm**

We exploit the fact that the projected circle is an ellipse. Thus, ellipse's
major axis is the radius of rotation. If we set the beam direction to be
the :math:`z`-axis :math:`(0,0,1)`, we can determine the :math:`xy` angle
:math:`(\phi)` by the major axis angle of the ellipse. Furthermore, based
on the ratio between minor and major axis, we can determine the projected
circle's inclination from its normal, yielding the :math:`zy` angle
:math:`(\psi)` of the circle. One then needs to rotate the the sample
:math:`(-\phi,\pm \psi)`. The :math:`\psi` angle ambiguity comes from
the fact that we are working with projected data.

There are three possible situations with the ellipse fitting algorithm:

* The sample does not move in the image sequence, it is not off-centered
    enough.
* The ellipse degenerates to a line, which means the circle is parallel
    to the beam, thus we only need to rotate by :math:`-\phi`.
* The projections lead to a non-degenerate ellipse. In this case we need to
    rotate around both angles.

**Input**

An image sequence of at least 5 images with different sample positions
around the axis of rotation. It is highly recommended to use more images
in order to obtain more precise results from noisy data.

**Output**

:math:`\phi` and :math:`\psi` angles.

**Steps**

The measuring procedure is done in the following steps:

#. Sample extraction.
#. Sample tip determination (largest Euclidean distance from the sample
    entering to the image and its boundaries).
#. Ellipse fitting to the set of tip points.
#. Determine :math:`\phi` and :math:`\psi` from the projected circle.

**Asumptions**

The procedure assumes a high-absorbing sample with conical shape. Thanks to
such sample the segmentation procedure can be simplified to a thresholding
technique.

"""
import numpy as np
from scipy import ndimage
from concert.quantities import q


class Ellipse(object):

    """Ellipse fitting from a set of data points."""

    def __init__(self, images=None):
        self._images = images
        # XY angle.
        self._phi = None
        # ZY angle.
        self._psi = None
        # Ellipse center.
        self._center = None
        # Ellipse parameters.
        self._params = None
        # Sample tips
        self._tips = None

    def __call__(self):
        if self._images is None:
            raise RuntimeError("Images have not been set.")
        if self._phi is None or self._psi is None:
            self._determine_angles()
        return self._phi, self._psi

    @property
    def images(self):
        """Images of the rotated sample."""
        return self._images

    @images.setter
    def images(self, images):
        """Set image sequence to *images*."""
        self._phi = None
        self._psi = None
        self._center = None
        self._params = None
        self._tips = None
        self._images = images

    @property
    def center(self):
        """Ellipse center (y, x)."""
        if self._center is None:
            self._determine_center()

        return self._center

    @property
    def xy_angle(self):
        """Angle between x and y axes :math:`(\phi)`."""
        if self._phi is None:
            self._determine_angles()
        return self._phi

    @property
    def zy_angle(self):
        """Angle between z and y axes :math:`(\psi)`."""
        if self._psi is None:
            self._determine_angles()
        return self._psi

    @property
    def points(self):
        """Return sample tip points."""
        if self._tips is None:
            self._determine_tips()

        return self._tips

    def _fit_ellipse(self):
        """Ellipse fitting based on *points* and singular value
        decomposition.
        """
        if self._tips is None:
            self._determine_tips()

        a_matrix = _construct_matrix(self._tips)
        v_mat = np.linalg.svd(a_matrix)[2]
        self._params = v_mat.T[:, -1]

    def _determine_tips(self):
        """Get sample tips from images."""
        if self._tips is None:
            images = [_segment(image) for image in self._images]
            self._tips = _get_ellipse_points(images)
            if len(self._tips) == 0:
                raise ValueError("No sample tip points found.")

    def _determine_center(self):
        """Determine the center of the ellipse from its parameters."""
        if self._params is None:
            self._fit_ellipse()

        a_33 = np.array([[self._params[0], self._params[1] / 2],
                        [self._params[1] / 2, self._params[2]]])
        if np.linalg.det(a_33) > 0:
            x_pos = (self._params[1] * self._params[4] -
                     2 * self._params[3] * self._params[2]) /\
                (4 * self._params[0] * self._params[2] - self._params[1] ** 2)
            y_pos = (self._params[3] * self._params[1] -
                     2 * self._params[0] * self._params[4]) /\
                (4 * self._params[0] * self._params[2] - self._params[1] ** 2)
        else:
            # We are not dealing with an ellipse, use center of mass.
            y_pos, x_pos = center_of_mass(self._tips)

        # +1 for image edge-clipping compensation.
        self._center = y_pos + 1, x_pos + 1

    def _determine_angles(self):
        """
        Determine the :math:`xy` :math:`(\phi)` and :math:`zy` angle
        :math:`(\psi)` of the normal of the circle defined by the axis
        of rotation. The axis is determined by fitting an ellipse
        to the sample rotation visible by different tips positions
        in images. Return a tuple :math:`(\phi`, :math:`\psi)`.
        """
        if self._params is None:
            self._fit_ellipse()

        y_ind, x_ind = zip(*self._tips)
        x_ind = np.array(x_ind)
        y_ind = np.array(y_ind)
        d_x = x_ind.max() - x_ind.min()
        if d_x < 10:
            raise ValueError("Sample off-centering too small, " +
                             "enlarge rotation radius.")

        a_33 = np.array([[self._params[0], self._params[1] / 2],
                         [self._params[1] / 2, self._params[2]]])

        usv = np.linalg.svd(a_33)
        s_vec = usv[1]
        v_mat = usv[2]

        if np.linalg.det(a_33) <= 0:
            # Not an ellipse
            d_y = float(y_ind.max() - y_ind.min())
            angle = np.arctan(d_y / d_x) * q.rad
            if x_ind[y_ind.argmax()] <= x_ind[y_ind.argmin()]:
                # More than pi/4.
                angle = -angle
            self._phi, self._psi = angle, 0.0 * q.rad
        else:
            self._phi, self._psi = \
                np.arctan(v_mat[1][1] / v_mat[1][0]) * q.rad, \
                np.arcsin(np.sqrt(s_vec[1]) / np.sqrt(s_vec[0])) * q.rad


def _construct_matrix(points):
    """Construct a conical section matrix from data *points*."""
    y_ind, x_ind = zip(*points)
    matrix = np.empty((len(x_ind), 6), dtype=np.int)
    for i in range(len(x_ind)):
        matrix[i] = np.array([x_ind[i] ** 2, x_ind[i] * y_ind[i],
                              y_ind[i] ** 2, x_ind[i], y_ind[i], 1])

    return matrix


def _segment(image, k=3.0):
    """Segment an *image* into a sample and a background. Suppose the low
    intensities caused by a highly absorbing sample are located below 1/*k*
    of the maximum intensity in the image.
    """
    bins = np.histogram(image, 128)[1]
    if bins[0] > bins[-1] / k:
        # Minimum intensity is too high, sample out of the FOV.
        thr = 0
    else:
        thr = (bins[-1] - bins[0]) / k + bins[0]

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
    tip_center = center_of_mass(tips)

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
    y_ind = y_ind[above]
    x_ind = x_ind[above]

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


def _get_ellipse_points(images):
    """Extract ellipse points from the sample in *image*, where *image* is
    a binary image."""
    tips = []
    tip = None

    for image in images:
        labels = _get_sample(image)
        tip = _get_sample_tip(labels)
        if tip is not None:
            tips.append(tip)

    return tips


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
    mask = sizes < 200
    remove_indices = mask[labels]
    labels[remove_indices] = 0

    return labels


def center_of_mass(points):
    """Find the center of mass from a set of *points*."""
    y_ind, x_ind = zip(*points)

    c_y = float(np.sum(y_ind)) / len(points)
    c_x = float(np.sum(x_ind)) / len(points)

    return c_y, c_x
