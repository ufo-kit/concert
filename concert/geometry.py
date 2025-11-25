"""Module for space transformations."""
import numpy as np
from concert.quantities import q


X_AX = np.array([1, 0, 0]) * q.dimensionless
Y_AX = np.array([0, 1, 0]) * q.dimensionless
Z_AX = np.array([0, 0, 1]) * q.dimensionless


def get_vector_length(vector):
    """Get length of a *vector*."""
    return np.sqrt(np.sum(vector ** 2, axis=0))


def normalize(vector):
    """Normalize a *vector*, result is unitless."""
    assert get_vector_length(vector).magnitude != 0

    return (vector / get_vector_length(vector)).to_base_units().magnitude


def translate(vec):
    """Translate the object by a vector *vec*. The vector is _always_ transformed into meters and
    the resulting transformation matrix is unitless.
    """
    vec = vec.to(q.m).magnitude
    trans_matrix = np.identity(4)

    trans_matrix[0][3] = vec[0]
    trans_matrix[1][3] = vec[1]
    trans_matrix[2][3] = vec[2]

    return trans_matrix


def rotate(phi, axis, shift=None):
    """Rotate the object by *phi* around vector *axis*, where *shift* is the translation which takes
    place before the rotation and -*shift* takes place afterward, resulting in the transformation
    TRT^-1. Rotation around an arbitrary point in space can be modeled in this way. The angle is
    _always_ rescaled to radians. The resulting transformation matrix is unitless.
    """
    axis = normalize(axis)

    phi = phi.to(q.rad).magnitude
    sin = np.sin(phi)
    cos = np.cos(phi)
    v_x = axis[0]
    v_y = axis[1]
    v_z = axis[2]

    if shift is not None:
        t_1 = translate(shift)

    rot_matrix = np.identity(4)
    rot_matrix[0][0] = cos + v_x ** 2 * (1 - cos)
    rot_matrix[0][1] = v_x * v_y * (1 - cos) - v_z * sin
    rot_matrix[0][2] = v_x * v_z * (1 - cos) + v_y * sin
    rot_matrix[1][0] = v_x * v_y * (1 - cos) + v_z * sin
    rot_matrix[1][1] = cos + v_y ** 2 * (1 - cos)
    rot_matrix[1][2] = v_y * v_z * (1 - cos) - v_x * sin
    rot_matrix[2][0] = v_z * v_x * (1 - cos) - v_y * sin
    rot_matrix[2][1] = v_z * v_y * (1 - cos) + v_x * sin
    rot_matrix[2][2] = cos + v_z ** 2 * (1 - cos)

    if shift is not None:
        t_2 = translate(-shift)
        rot_matrix = np.dot(np.dot(t_1, rot_matrix), t_2)

    return rot_matrix
