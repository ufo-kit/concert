import numpy as np
from concert.quantities import q
from concert.devices.cameras.dummy import Base as DummyBaseCamera


def get_np_angle(angle):
    return np.array([angle.to(q.rad).magnitude] * q.rad)


def rot_x(angle, matrix):
    """Rotation around x-axis."""
    angle = get_np_angle(angle)
    m_0 = np.array([[1, 0, 0],
                    [0, np.cos(angle), -np.sin(angle)],
                    [0, np.sin(angle), np.cos(angle)]])
    new_mat = np.identity(4, np.float)
    new_mat[:3, :3] = m_0

    return np.dot(new_mat, matrix)


def rot_y(angle, matrix):
    """Rotation around y-axis."""
    angle = get_np_angle(angle)
    m_0 = np.array([[np.cos(angle), 0, np.sin(angle)],
                    [0, 1, 0],
                    [-np.sin(angle), 0, np.cos(angle)]])
    new_mat = np.identity(4, np.float)
    new_mat[:3, :3] = m_0

    return np.dot(new_mat, matrix)


def rot_z(angle, matrix):
    """Rotation around z-axis."""
    angle = get_np_angle(angle)
    m_0 = np.array([[np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1]])
    new_mat = np.identity(4, np.float)
    new_mat[:3, :3] = m_0

    return np.dot(new_mat, matrix)


def scale(scales, matrix):
    """Scaling amongst all directions."""
    scales = scales + (1,)

    return np.dot(np.diag(scales), matrix)


def translate(vec, matrix):
    """3D translation."""
    matrix[0, -1] += vec[0]
    matrix[1, -1] += vec[1]
    matrix[2, -1] += vec[2]

    return matrix


def sphere(size, radius, mat):
    """Create a sphere with radius *radius* in an image with size *size* and
    use transformation matrix *mat* to adjust the sphere position and size.
    """
    y_0, x_0 = np.mgrid[-size / 2:size / 2, -size / 2:size / 2]

    k_x = x_0 * mat[0, 0] + y_0 * mat[0, 1] + mat[0, 3]
    k_y = x_0 * mat[1, 0] + y_0 * mat[1, 1] + mat[1, 3]
    k_z = x_0 * mat[2, 0] + y_0 * mat[2, 1] + mat[2, 3]

    quadr_coeffs = mat[0, 2] ** 2 + mat[1, 2] ** 2 + mat[2, 2] ** 2, \
        2 * k_x * mat[0, 2] + 2 * k_y * mat[1, 2] + 2 * k_z * mat[2, 2], \
        k_x ** 2 + k_y ** 2 + k_z ** 2 - radius ** 2

    thickness = quadr_coeffs[1] ** 2 - 4 * quadr_coeffs[0] * quadr_coeffs[2]
    thickness[thickness < 0] = 0

    return np.abs(np.sqrt(thickness) / quadr_coeffs[0])


def transfer(thickness, ref_index, lam):
    """Beer-Lambert law."""
    lam = lam.to(thickness.units)
    mju = 4 * q.dimensionless * np.pi * ref_index.imag / lam

    return np.exp(- mju * thickness)


def get_projection(thickness):
    """Get X-ray projection image from projected thickness."""
    max_val = thickness.max()
    if max_val > 0:
        thickness = thickness / thickness.max()

    # Iron and 20 keV
    ref_index = 3.85263274e-06 + 9.68238822e-08j
    lam = 6.1992e-11 * q.m

    # Do not take noise into account in order to make test results
    # reproducible.
    return transfer(thickness * q.mm, ref_index, lam)


class SimulationCamera(DummyBaseCamera):

    """A dummy image source providing images of a rotated sample. Rotation
    is based on virtual motors.
    """
    ITER = "iteration"

    def __init__(self, im_width, x_param, y_param, z_param,
                 needle_radius=None, rotation_radius=None,
                 y_position=None, scales=None):
        self.x_axis_param = x_param
        self.y_axis_param = y_param
        self.z_axis_param = z_param
        self.size = im_width
        self._center = None
        self.radius = needle_radius
        if self.radius is None:
            self.radius = self.size / 8

        self.rotation_radius = rotation_radius if rotation_radius is not None\
            else self.size / 6
        self.y_position = y_position if y_position \
            is not None else self.size / 2
        self.scale = scales if scales is not None else (3, 0.75, 3)

        # How many times was the image source asked for images.
        self.iteration = 0

        super(SimulationCamera, self).__init__()

    @property
    def ellipse_center(self):
        """Ellipse center."""
        return self._center

    def create_needle(self):
        """Create sample rotated about axis of rotation."""
        matrix = np.identity(4, np.float)

        matrix = translate((0, self.y_position, 0), matrix)

        matrix = rot_z(self.z_axis_param.get().result(), matrix)
        matrix = rot_x(self.x_axis_param.get().result(), matrix)

        center = np.dot(np.linalg.inv(matrix),
                        (0, self.size / 8 / self.scale[1], 0, 1)) + \
            self.size / 2
        # Ellipse center.
        self._center = center[1], center[0]

        matrix = rot_y(self.y_axis_param.get().result(), matrix)
        matrix = translate((self.rotation_radius, 0, 0), matrix)

        matrix = scale(self.scale, matrix)

        return sphere(self.size, self.radius, matrix)

    def _grab_real(self):
        return get_projection(self.create_needle())
