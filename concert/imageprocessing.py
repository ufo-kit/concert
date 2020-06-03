"""
Image processing module for manipulating image data, e.g. filtered
backprojection, flat field correction and other operations on images.
"""

import numpy as np
import logging
from scipy.signal import fftconvolve
from concert.quantities import q


LOG = logging.getLogger(__name__)


def normalize(image, minimum=0.0, maximum=1.0):
    """Normalize *image* intensities to start at *minimum* and end at *maximum*."""
    mul = (float(maximum) - minimum) / (image.max() - image.min())

    return mul * (image - image.min()) + minimum


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


def find_needle_tips(images):
    """Get sample tips from images."""
    tips = []

    for image in images:
        tip = find_needle_tip(image)
        if tip is not None:
            tips.append(tip)

    LOG.debug('Needle tips: %s', np.array(tips).tolist())

    if len(tips) == 0:
        raise ValueError("No sample tip points found.")

    return tips


def find_needle_tip(image):
    """Extract needle tip from *image*."""
    mask = segment_convex_object(image)
    if mask is None:
        return None
    coords = np.array(list(zip(*np.where(mask))))
    min_y = np.min(coords[:, 0])
    indices = np.where(coords[:, 0] == min_y)[0]
    coords = coords[indices]
    if coords[:, 1].max() - coords[:, 1].min() > image.shape[1] // 4:
        # Needle tip cannot be width / 4 broad, we have probably segmented just noise
        return None
    coords = [_find_peak_subpix(pos, image) for pos in coords]

    return np.mean(coords, axis=0) if coords else None


def segment_convex_object(image):
    """
    Extract convex object from *image* (e.g. needle or sphere). It doesn't matter if object is
    brigher or darker than the background (e.g. non flat corrected radiograph on input).
    """
    try:
        from skimage.filters import threshold_otsu
        from skimage.morphology import convex_hull_image, label
    except ImportError as e:
        print "You need to install scikit-image in order to use this function"
        LOG.error(e)

    def _segment(threshold, greater=True):
        mask = np.zeros_like(image, dtype=np.int8)
        if greater:
            mask[image > threshold] = 1
        else:
            mask[image < threshold] = 1
        labels, num = label(mask, return_num=True)
        bins = np.arange(num + 1) + 0.5
        hist, bins = np.histogram(labels, bins=bins)
        largest_label = int(bins[np.argmax(hist)] + 0.5)
        mask[labels != largest_label] = 0

        return mask

    try:
        thr_otsu = threshold_otsu(image)
    except Exception as e:
        LOG.error(e)
        return None

    sgn = 1
    mask = _segment(thr_otsu)

    # Compute convex hull of the mask and inverted mask. Object is that mask which has smaller
    # amount of pixels added by the convex hull (convex hull of a convex polygon has the same size
    # as the polygon itself, whereas for concave polygons we'd need to add some pixels to the hull).
    # Since the object segmentation can be jagged, convex hull might still add some pixels, so don't
    # test for 0 but for the difference between added pixels for mask and inverted mask.
    imask = 1 - mask
    hull = convex_hull_image(mask)
    ihull = convex_hull_image(imask)
    hull_diff = np.count_nonzero(hull) - float(np.count_nonzero(mask))
    ihull_diff = np.count_nonzero(ihull) - float(np.count_nonzero(imask))
    if hull_diff > ihull_diff:
        mask = imask
        sgn = -1

    # Refine the segmentation, set the threshold to roughly FWTM of the background standard
    # deviation.
    indices = np.where(1 - mask)
    mean_bg = image[indices].mean()
    std_bg = image[indices].std()
    thr = mean_bg + sgn * 5 * std_bg
    if sgn == 1 and thr < thr_otsu or sgn == -1 and thr > thr_otsu:
        mask = _segment(thr, greater=sgn == 1)

    return mask


def _find_peak_subpix(peak, image, supersampling=16):
    """Supersample vertical line at the *peak* (y, x) position by *supersampling* and look for the
    steepest gradient in the region (peak[0] - 1, peak[0] + 1) in the high resolution line.
    """
    from scipy.ndimage import gaussian_filter1d
    dy = 8
    y_start = max(peak[0] - dy, 0)
    line = image[y_start:min(peak[0] + dy, image.shape[0]), peak[1]]
    x = np.arange(len(line))
    x_hd = np.arange(0, len(line) - 1 + 1. / supersampling, 1. / supersampling)
    line_hd = np.interp(x_hd, x, line)
    # FWHM of the low resolution pixel
    sigma = supersampling / (2. * np.sqrt(2 * np.log(2)))
    blurred = gaussian_filter1d(line_hd, sigma)
    middle = len(x) * supersampling // 2
    g = np.abs(np.gradient(blurred))[middle - supersampling:middle + supersampling + 1]
    y = (np.argmax(g) + middle - supersampling) / float(supersampling) + y_start

    return (y, peak[1])


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


def filter_low_frequencies(data, fwhm=32.):
    """Filter low frequencies in 1D *data*. *fwhm* is the FWHM of the gaussian used to filter out
    low frequencies in real space. The window is then computed as fft(1 - gauss).
    """
    mean = np.mean(data)
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    # We compute the gaussian in Fourier space, so convert sigma first
    f_sigma = 1. / (2 * np.pi * sigma)
    x = np.fft.fftfreq(len(data))
    fltr = 1 - np.exp(- x ** 2 / (2 * f_sigma ** 2))

    return np.fft.ifft(np.fft.fft(data) * fltr).real + mean
