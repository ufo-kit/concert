"""
Measures operate on some data and provide information about how \"good\"
the data is.
"""
import numpy as np
from concert.quantities import q
from concert.imageprocessing import center_of_mass, get_needle_tips


class DummyGradientMeasure(object):

    """Gradient measure that returns a quadratic fall-off of *parameter* from
    *max_position*."""

    def __init__(self, parameter, max_position):
        self.max_position = max_position
        self._max_gradient = 1e4
        self._param = parameter

    def __call__(self):
        value = self._param.get().result()
        position = value.to(self.max_position.units).magnitude
        return self._max_gradient - \
            (position - self.max_position.magnitude) ** 2


class Area(object):

    """
    Area measures how much of the whole image is covered by an object.
    The return value is the fraction between object pixels and the whole
    area, :math:`n = object\_pixels / (width \cdot height)`.

    Parameters are:

    .. py:attribute:: func

        a callable *func(radio, \*args, \*\*kwargs)*, where
        *radio* is a radiograph to be analyzed, *args* and *kwargs*
        are additional positional and keyword arguments passed to *func*
        which returns the number of pixels covered by
        an object

    .. py:attribute:: args

        *func* arguments tuple

    .. py:attribute:: kwargs

        *func* keyword arguments dictionary

    """

    def __init__(self, func, args=None, kwargs=None):
        self.func = func
        self.args = () if args is None else args
        self.kwargs = {} if kwargs is None else kwargs

    def __call__(self, radio):
        object_pixels = self.func(radio, *self.args, **self.kwargs)

        return float(object_pixels) / (radio.shape[0] * radio.shape[1])


class SimpleArea(Area):

    """
    An area measure with object segmentation by a threshold. If two
    flat fields *flat_1* and *flat_2* are given, then the minimum
    of their subtraction :math:`min(flat\_1, flat\_2)` approximates
    the maximum grey value considered a background. The values
    from the :math:`radiograph - flat` below the minimum are
    considered an object.
    """

    def __init__(self, flat_1, flat_2):
        super(SimpleArea, self).__init__(self.get_object_pixels)
        self._flat_1 = flat_1
        self._flat_2 = flat_2
        self.flat_avg = None
        self.threshold = None
        self._setup()

    def get_object_pixels(self, radio):
        """Get object pixels from a radiograph *radio*."""
        radio = radio - self.flat_avg

        return len(np.where(radio < self.threshold)[0])

    def set_flats(self, flat_1, flat_2):
        """
        Set new flat fields *flat_1*, *flat_2*. The threshold and
        average flat are recomputed.
        """
        if flat_1.shape != flat_2.shape:
            raise ValueError("Flats must have the same shape")
        self._flat_1 = flat_1
        self._flat_2 = flat_2
        self._setup()

    @property
    def flats(self):
        """flat_1 used by measure"""
        return [self._flat_1, self._flat_2]

    def _setup(self):
        """Setup the threshold and average flat field."""
        # Threshold
        self.threshold = np.min(self._flat_1 - self._flat_2)
        # Average the two flats to suppress the noise a little
        self.flat_avg = (self._flat_1 + self._flat_2) / 2.0


EPSILON = 1e-3


class Ellipse(object):

    """
    Determine a 3D circle normal inclination angles towards the
    :math:`y`-axis (0,1,0) and the center of the circle.

    **Algorithm**

    We exploit the fact that the projected circle is an ellipse. Thus,
    ellipse's major axis is the radius of rotation. If we set the beam
    direction to be the :math:`z`-axis :math:`(0,0,1)`, we can determine
    the :math:`xy` angle :math:`(\phi)` by the major axis angle
    of the ellipse. Furthermore, based on the ratio between minor
    and major axis, we can determine the projected circle's
    inclination from its normal, yielding the :math:`zy` angle
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
            self._tips = get_needle_tips(self.images)

        return self._tips

    def _fit_ellipse(self):
        """Ellipse fitting based on *points* and singular value
        decomposition.
        """
        if self._tips is None:
            self._tips = get_needle_tips(self.images)

        a_matrix = _construct_matrix(self._tips)
        v_mat = np.linalg.svd(a_matrix)[2]
        self._params = v_mat.T[:, -1]

    def _determine_center(self):
        """Determine the center of the ellipse from its parameters."""
        if self._params is None:
            self._fit_ellipse()

        a_33 = np.array([[self._params[0], self._params[1] / 2],
                        [self._params[1] / 2, self._params[2]]])
        if np.linalg.det(a_33) > EPSILON:
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
