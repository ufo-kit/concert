"""
Measures operate on some data and provide information about how \"good\"
the data is.
"""
import logging
from typing import List, Tuple
import numpy as np
import skimage.measure as skm
from concert.quantities import q, Quantity
from concert.typing import ArrayLike


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


def estimate_alignment_parameters(centroids: List[ArrayLike], min_data_pts: int = 5,
                                  x_offset_px: int = 10, use_svd: bool = False
                                  ) -> Tuple[Quantity, Quantity, float]:
    """
    Estimates roll, pitch and center of rotation from projected sphere phantom.

    This implementation incorporates more robust methods to determine the degenerate case of ellipse
    using `skimage.measure.EllipseModel`, as opposed to SVD. The downside of using SVD throughout
    the estimation is that the ellipse parameters computed from SVD in least-square sense are
    very small floating point values in practice and they get even smaller during matrix operations.
    Comparing among these values is fragile.

    To determine the nature of the ellipse-fit from sphere centroids we feed them to EllipseModel,
    which encapsulates direct least-square fit incorporating the conic discriminant condition
    B^2 - 4AC < 0. In other words, unless the pitch angle is exactly 0 degree we are bound to get an
    ellipse and its parameters, center-x, center-y, length of semi-major axis, length of semi-minor
    axis, roll angle of the major axis in radians. If the length of semi-minor axis is greater or
    equal to 1 pixel we have a proper ellipse and degenerate case otherwise.

    - If a proper ellipse is found we can optionally make use of SVD to find the center of the
    ellipse. A design matrix is built using the sphere centroids, where each row vector is
    (x_i^2, x_iy_i, y_i^2, x_i, y_i, 1), i in range(#centroids). SVD of design matrix yields
    best-fit parameters A, B, C, D, E, F in least-square sense. Center of the ellipse is the
    (x, y) point for which the gradient of the quadratic form of the conic (degree-2 polynomial
    terms) of Ax^2 + Bxy + Cy^2 + Dx + Ey + F becomes 0. Taking the gradient and solving the
    same for (x, y) using Cramer's rule the center is computed as,

     - x_c = (BF - 2DC) / (4AC - B^2)
     - y_c = (DB - 2AE) / (4AC - B^2)

    Alternative to using SVD, we can use the centers from the best-fit parameters from
    EllipseModel directly. Upon computing the center (x, y) we compute two direction vectors
    out of the sphere centroids. Dot product of these vectors provides the cosine component and
    cross product provides the sine component along with the signed area of the parallelogram
    that they make depending on their orientation w.r.t. each other, which is helpful to
    determine the rotation direction of the pitch using arctangent. If using SVD we compute the
    magnitude of the pitch from ratio of square-roots of the singual values of the quadratic
    form matrix [[A, B / 2], [B / 2, C]], which is equivalent of ratio of semi-minor and
    semi-major axes. Axes lengths and roll angle can be directly obtained from EllipseModel.

    - Upon encountering degenerate ellipse case pitch angle can be inferred as 0 degree. Roll angle
    is determined using slope (rise / run) with arctangent function. In this case an extra check
    for the rotation of major-axis is done to determine the correct roll angle rotation.

    In both cases of ellipse a normalization of the estimated roll angles is done to limit it to
    a conveninent range. Experiments show that fitting noise can affect the roll-angle estimation.

    :param centroids: extracted sphere centroids from projections
    :type centroids: List[ArrayLike]
    :param min_data_pts: min #centroids to consider to avoid an under-determined system
    :type min_data_pts: int
    :param x_offset_px: equivalent of min rotation radius in #pixels to consider for good estimation
    :type x_offset_px: int
    :param use_svd: if estimation should be done using SVD
    :type use_svd: bool
    :return: estimated roll, pitch and center of rotation
    :rtype: Tuple[`concert.quantities.Quantity`, `concert.quantities.Quantity`, float]
    """
    
    def _normalized(angle: float) -> float:
        """
        Rotating the major axis by 180 degrees produces the same ellipse. For instance when actual
        roll angle is approximately 0 degree, fitting noise can make the angle flip to approximately
        180 degrees instead. We map the estimated angle to a smaller range of (-90°, 90°].
        """
        wrapped = (angle + np.pi) % (2 * np.pi) - np.pi
        if wrapped > np.pi/2:
            wrapped -= np.pi
        elif wrapped <= -np.pi/2:
            wrapped += np.pi
        return wrapped

    if len(centroids) < min_data_pts:
        raise ValueError("under-determined system, at least 5 coordinate pairs are needed")
    y_ind, x_ind = np.array(centroids)[:, 0], np.array(centroids)[:, 1]
    if max(x_ind) - min(x_ind) < x_offset_px:
        raise ValueError("sample off-centering too small, enlarge rotation radius")
    # We fit the projection centroids to the EllipseModel to determine if we have a proper ellipse.
    elp = skm.EllipseModel()
    is_ellipse = elp.estimate(data=np.array(centroids)[:, [1, 0]])
    if is_ellipse:
        xc, yc, semi_major, semi_minor, roll_angle_radians = elp.params
    if is_ellipse and np.round(semi_minor).astype(int) >= 1:  # Proper Ellipse
        if use_svd:
            # Construct design matrix - system of equations from which we want to estimate the
            # quadratic and linear parameters that best fit the ellipse. General equation of conic
            # has six parameters, A B, C, D, E, F. Therefore design matrix has #data_points rows and
            # six columns.
            design_matrix = np.empty((len(x_ind), 6), dtype=float)
            for i in range(len(x_ind)):
                design_matrix[i] = np.array([x_ind[i] ** 2, x_ind[i] * y_ind[i],
                                            y_ind[i] ** 2, x_ind[i], y_ind[i], 1])
            # Center of a conic (Ellipse or Hyperbola) is the point where the gradient of the
            # quadratic form becomes 0. To compute this (x, y) points we take the gradients of the
            # general conic equation and make a linear system of equations out it. Upon solving the
            # same using Cramer's rule we'd get the exact expressions for x_center and y_center
            # below. Best-fit parameters come from the SVD of the design matrix above.
            _, _, parmas_matrix = np.linalg.svd(design_matrix)
            params = parmas_matrix.T[:, -1]
            prm_A, prm_B, prm_C = params[0], params[1], params[2]
            prm_D, prm_E = params[3], params[4]
            x_center = ((prm_B * prm_E - 2 * prm_D * prm_C) / (4 * prm_A * prm_C - prm_B ** 2))
            y_center = ((prm_D * prm_B - 2 * prm_A * prm_E) / (4 * prm_A * prm_C - prm_B ** 2))
            center = np.array([y_center, x_center])
            # We take two points out of the sphere centroids which make the projected ellipse,
            # remove the pre-existing translations in physical space and get two direction vectors
            # from origin. These would be used to determine the sign (rotation direction) of the
            # pitch angle. We can compute the magnitude of the pitch from singular values but to get
            # the directional information we use dot product, cross product with inverse tangent.
            # Dot product provides the magnitude of, how two direction vectors are aligned with each
            # other and brings in the cosine component.
            # Cross product for the 2D direction vectors provide the area of the paralleogram that
            # they make and depending on the orientation of the vectors with respect to each other
            # it can be positive or negative. Here it is supposed to give us the hint, whether
            # dir_vec_1 is to the clockwise or anticlockwise direction from dir_vec_0 and it brings
            # in the sine component.
            dir_vec_0 = np.array(centroids[0]) - center
            dir_vec_1 = np.array(centroids[1]) - center
            pitch_d_angle = np.arctan2(np.cross(dir_vec_0, dir_vec_1), np.dot(dir_vec_0, dir_vec_1))
            sign = int(np.sign(pitch_d_angle))
            quadratic_form_matrix = np.array([[prm_A, prm_B / 2], [prm_B / 2, prm_C]])
            # Quadratic form matrix is symmetric. Therefore its singular values and singular vectors
            # are identical to eigen values and eigen vectors. But for consistency we name them
            # sing_vals and sign_vecs. Following SVD in principle has,
            # # first matrix = eigen vectors along semi-major and semi-minor axes as column vectors,
            # # second diagonal matrix = corresponding eigen values,
            # # third matrix = first matrix transposed i.e. same eigen vectors as row vectors.
            _, sing_vals, _ = np.linalg.svd(quadratic_form_matrix)
            # The ratio of sqrt(eigen values) along semi-minor and semi-major axes is proportional
            # to the ratio of semi-minor and semi-major axes. Normally eigen values and eigen
            # vectors are in decreasing order, hence sing_vals[0] is the eigen value along major
            # axis and sing_vals[1] is he eigen value along minor axis. In projected ellipse their
            # invserse sine gives us the magnitude of how much we are tilted. On top of that we
            # incorporate the direction.
            pitch_ang = sign * np.arcsin(np.sqrt(sing_vals[1]) / np.sqrt(sing_vals[0])) * q.rad
        else:
            # Instead of using SVD we use the optimal parameters from the least square ellipse fit.
            center = np.array([yc, xc])
            dir_vec_0 = np.array(centroids[0]) - center
            dir_vec_1 = np.array(centroids[1]) - center
            pitch_d_angle = np.arctan2(np.cross(dir_vec_0, dir_vec_1), np.dot(dir_vec_0, dir_vec_1))
            sign = int(np.sign(pitch_d_angle))
            pitch_ang = sign * np.arcsin(semi_minor / semi_major) * q.rad
        roll_ang = _normalized(roll_angle_radians) * q.rad
        rot_cnt = xc
    else:  # Degenerate Ellipse
        # In this scenario we make use of the slope (rise / run) and inverse tanget to derive the
        # roll angle.
        d_y = float(y_ind.max() - y_ind.min())
        d_x = float(x_ind.max() - x_ind.min())
        angle = _normalized(np.arctan2(d_y , d_x)) * q.rad
        # We determining the correct sign for the roll angle under the assumption that our
        # coordinate origin is at the top-left. Since y-axis denotes the vertical direction, if
        # following condition is satisfied it'd mean major-axis is rotated counter-clockwise and
        # it'd take a clockwise (-ve) angular rotation to correct that.
        if x_ind[y_ind.argmax()] <= x_ind[y_ind.argmin()]:
            angle = -angle
        roll_ang, pitch_ang = angle, 0.0 * q.rad
        rot_cnt = np.mean(x_ind)
    return roll_ang.to(q.deg), pitch_ang.to(q.deg), rot_cnt
