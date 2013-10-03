"""
Image processing module for manipulating image data, e.g. filtered
backprojection, flat field correction and other operations on images.
"""

import multiprocessing
import numpy as np
import logbook
from concert.helpers import threaded


LOG = logbook.Logger(__name__)


def flat_correct(radio, flat, dark=None):
    """
    Flat field correction of a radiograph *radio* with *flat* field.
    If *dark* field is supplied it is taken into account as well.
    """
    return radio / flat if dark is None else (radio - dark) / (flat - dark)


def backproject(sinogram, center, angle_step=None, start_projection=0,
                x_points=None, y_points=None, normalize=False, fast=True):
    """
    CT backprojection algorithm. Take a *sinogram* with *center* of
    rotation, *angle_step* the rotation angular step between two
    consecutive projections and reconstruct a slice. *start_projection*
    determines the index of the projection from the whole data set
    which is mapped to the first row of the *sinogram*. *x_points*
    and *y_points* are 2D arrays of the slice grid indices. *normalize*
    is True if the slice should be normalized based on inverse Radon
    transform and the number of projections. If *fast* is True, use
    a faster backprojection algorithm, which takes much more memory.

    *sinogram* does not have to be a complete one, it can be a continuous
    subset of angles of rotation, even only one row of the complete sinogram.
    In either case it must be a 2D array. If one wants to reconstruct
    a partial slice, *angle_step* and *start_projection* must be provided
    in order to correctly determine the angle of rotation. In case
    the whole slice should be reconstructed, those parameters can be skipped.

    If *angle_step* is None it is calculated from the number of
    projections, i.e. sinogram height. If *x_points* and *y_points*
    are None they are calculated from the slice width and center of
    rotation.

    """
    num_projections = sinogram.shape[0]
    width = sinogram.shape[1]

    # Initialize for case we want to reconstruct from the whole sinogram
    if angle_step is None:
        # Figure out the angle step on our own
        angle_step = np.pi / num_projections
    if x_points is None and y_points is None:
        y_points, x_points = np.mgrid[-center:width - center,
                                      -center:width - center]
        x_points = x_points.astype(np.float32)
        y_points = y_points.astype(np.float32)
    end_projection = num_projections + start_projection
    angles = angle_step * np.arange(start_projection, end_projection).\
        astype(np.float32)
    # Backprojection normalization
    norm = get_backprojection_norm(end_projection -
                                   start_projection) if normalize else 1

    def prepare_fast():
        """Prepare 3D index and trigonometric arrays for backprojection."""
        sin_angles = np.sin(angles)
        cos_angles = np.cos(angles)
        sin_angles_ext = sin_angles.repeat(width ** 2).reshape(
            end_projection - start_projection, width, width)
        cos_angles_ext = cos_angles.repeat(width ** 2).reshape(
            end_projection - start_projection, width, width)
        indices = np.arange(num_projections, dtype=np.int16).repeat(
            width ** 2).reshape(end_projection -
                                start_projection, width, width)

        return sin_angles_ext, cos_angles_ext, indices

    @threaded
    def backproject_slow(result, start, stop):
        """
        Classical backprojection by explicitly going
        through every projection.
        """
        for i in range(start_projection, end_projection):
            pos = x_points[start:stop, :] * np.sin(angles[i]) + \
                y_points[start:stop, :] * np.cos(angles[i]) + center
            pos[np.where((pos < 0) | (pos >= width))] = 0
            result[start:stop, :] += sinogram[i, pos.astype(np.int32)]

    # When this is @async the results are sometimes wrong -> TODO: investigate!
    @threaded
    def backproject_fast(result, start, stop):
        """Backproject part of the slice using numpy arrays."""
        # Get sinogram position in the slice
        x_pos = x_points[start:stop, :] * sin_angles_ext[:, start:stop, :]
        y_pos = y_points[start:stop, :] * cos_angles_ext[:, start:stop, :]
        pos = x_pos + y_pos + center
        # Simple nearest-neighbor interpolation
        pos[np.where((pos < 0) | (pos >= width))] = 0
        result[start:stop, :] = np.sum(sinogram[indices[:, start:stop, :],
                                                pos.astype(np.int32)],
                                       axis=0) * norm

    def execute(fast=fast):
        """Execute a backprojecting scheme based on *fast*."""
        backprojector = backproject_fast if fast else backproject_slow
        if fast:
            result = np.empty((width, width), dtype=np.float32)
        else:
            result = np.zeros((width, width), dtype=np.float32)

        threads = []
        num_threads = multiprocessing.cpu_count()
        step = width / num_threads
        for i in range(num_threads):
            end = (i + 1) * step if i < num_threads - 1 else width
            threads.append(backprojector(result, i * step, end))
        for thread in threads:
            thread.join()

        return result

    if fast:
        try:
            # Prepare the 3D arrays and try to execute the fast algorightm
            sin_angles_ext, cos_angles_ext, indices = prepare_fast()
            result = execute()
        except MemoryError:
            # If there is not enough memory for the fast algorithm, fall back
            # to the classical one
            message = "Insufficient memory, try fast=False"
            LOG.debug(message)
            # Re-raise, so user can react
            raise MemoryError(message)
    else:
        result = execute(fast=False)

    return result


def get_backprojection_norm(num_projections):
    """
    Get the normalization of backprojected slice based on *num_projections*.
    """
    return 1 / (2 * np.pi * num_projections)


def get_ramp_filter(width):
    """Get a 1D ramp filter for filtering sinogram rows."""
    base = np.arange(-width / 2, width / 2)

    return np.fft.fftshift(np.abs(base)) * 2.0 / width
