"""
Measures operate on some data and provide information about how \"good\"
the data is.
"""
import logging
import numpy as np
from concert.quantities import q


LOG = logging.getLogger(__name__)


class DummyGradientMeasure(object):

    """Gradient measure that returns a quadratic fall-off of *parameter* from
    *max_position*."""

    def __init__(self, parameter, max_position, negative=False):
        self.max_position = max_position
        self.negative = negative
        self._max_gradient = 1e4
        self._param = parameter

    async def __call__(self):
        value = await self._param.get()
        position = value.to(self.max_position.units).magnitude
        result = self._max_gradient - (position - self.max_position.magnitude) ** 2

        return -result if self.negative else result


class Area(object):

    r"""
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

        return object_pixels / (radio.shape[0] * radio.shape[1])


class SimpleArea(Area):

    r"""
    An area measure with object segmentation by a threshold. If two
    flat fields *flat_1* and *flat_2* are given, then the minimum
    of their subtraction :math:`min(flat\_1, flat\_2)` approximates
    the maximum grey value considered a background. The values
    from the :math:`radiograph - flat` below the minimum are
    considered an object.
    """

    def __init__(self, flat_1, flat_2):
        super(SimpleArea, self).__init__(self.object_pixels)
        self._flat_1 = flat_1
        self._flat_2 = flat_2
        self.flat_avg = None
        self.threshold = None
        self._setup()

    def object_pixels(self, radio):
        """Get object pixels from a radiograph *radio*."""
        radio = radio - self.flat_avg

        return len(np.where(radio < self.threshold)[0])

    @property
    def flats(self):
        """flat_1 used by measure"""
        return [self._flat_1, self._flat_2]

    @flats.setter
    def flats(self, flats):
        """
        Set new flat fields flat_1 and flat_2 from the tuple *flats*.
        The threshold and average flat are recomputed.
        """
        if flats[0].shape != flats[1].shape:
            raise ValueError("Flats must have the same shape")
        self._flat_1 = flats[0]
        self._flat_2 = flats[1]
        self._setup()

    def _setup(self):
        """Setup the threshold and average flat field."""
        # Threshold
        self.threshold = np.min(self._flat_1 - self._flat_2)
        # Average the two flats to suppress the noise a little
        self.flat_avg = (self._flat_1 + self._flat_2) / 2


def rotation_axis(tips):
    r"""
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

    **Steps**

    The measuring procedure is done in the following steps:

    #. Sample extraction.
    #. Sample tip determination (largest Euclidean distance from the sample
        entering to the image and its boundaries).
    #. Ellipse fitting to the set of tip points.
    #. Determine :math:`\phi` and :math:`\psi` from the projected circle.

    **Assumptions**

    The procedure assumes a high-absorbing sample with conic shape. Thanks to
    such sample the segmentation procedure can be simplified to a thresholding
    technique.

    **Output**

    The measure returns a tuple (:math:`\phi`, :math:`\psi`, center).
    """
    if len(tips) < 5:
        raise ValueError("At least 5 coordinate pairs are needed")

    y_ind, x_ind = list(zip(*tips))
    if max(x_ind) - min(x_ind) < 10:
        raise ValueError("Sample off-centering too small, enlarge rotation radius.")

    a_matrix = np.empty((len(x_ind), 6), dtype=float)
    for i in range(len(x_ind)):
        a_matrix[i] = np.array([x_ind[i] ** 2, x_ind[i] * y_ind[i],
                                y_ind[i] ** 2, x_ind[i], y_ind[i], 1])

    v_mat = np.linalg.svd(a_matrix)[2]
    params = v_mat.T[:, -1]
    a_33 = np.array([[params[0], params[1] / 2], [params[1] / 2, params[2]]])
    x_ind = np.array(x_ind)
    y_ind = np.array(y_ind)
    d_x = x_ind.max() - x_ind.min()
    usv = np.linalg.svd(a_33)
    s_vec = usv[1]
    v_mat = usv[2]
    det = np.linalg.det(a_33)
    LOG.debug('Conic determinant: %g', det)

    if det <= np.finfo(float).resolution:
        # Not an ellipse
        d_y = float(y_ind.max() - y_ind.min())
        angle = np.arctan(d_y / d_x) * q.rad
        if x_ind[y_ind.argmax()] <= x_ind[y_ind.argmin()]:
            # More than pi/4.
            angle = -angle
        phi, psi = (angle, 0.0 * q.rad)
        center = np.mean(tips, axis=0)
    else:
        x_pos = ((params[1] * params[4] - 2 * params[3] * params[2])
                 / (4 * params[0] * params[2] - params[1] ** 2))
        y_pos = ((params[3] * params[1] - 2 * params[0] * params[4])
                 / (4 * params[0] * params[2] - params[1] ** 2))
        center = np.array((y_pos, x_pos))
        # Determine rotation direction needed for the pitch angle
        v_0 = np.array(tips[0]) - center
        v_1 = np.array(tips[1]) - center
        pitch_d_angle = np.arctan2(np.cross(v_0, v_1), np.dot(v_0, v_1))
        sgn = int(np.sign(pitch_d_angle))
        phi, psi = (np.arctan(v_mat[1][1] / v_mat[1][0]) * q.rad,
                    sgn * np.arcsin(np.sqrt(s_vec[1]) / np.sqrt(s_vec[0])) * q.rad)

    phi = phi.to(q.deg)
    psi = psi.to(q.deg)

    LOG.debug('Found z angle: %s, x angle: %s, center: %s', phi, psi, center)

    return (phi, psi, center)
