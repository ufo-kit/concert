"""
Image processing module for manipulating image data, e.g. filtered
backprojection, flat field correction and other operations on images.
"""

import asyncio
import numpy as np
import logging
from typing import Dict, Tuple, Union

try:
    import cupy as xp
    import cupy.fft as xft
    import cupyx.scipy.signal.windows as xp_windows

    _has_cupy = True
except ModuleNotFoundError:
    print("cupy not available, defaulting to numpy")
    import numpy as xp
    import numpy.fft as xft
    from scipy.signal.windows import tukey as xp_windows

    _has_cupy = False
from scipy.signal import fftconvolve
from concert.coroutines.base import background, run_in_executor
from concert.quantities import q
from concert.typing import ArrayLike

LOG = logging.getLogger(__name__)


def normalize(image, minimum=0.0, maximum=1.0):
    """Normalize *image* intensities to start at *minimum* and end at *maximum*."""
    mul = (maximum - minimum) / (image.max() - image.min())

    return mul * (image - image.min()) + minimum


def flat_correct(radio, flat, dark=None):
    """
    Flat field correction of a radiograph *radio* with *flat* field.
    If *dark* field is supplied it is taken into account as well.
    """
    if dark is not None:
        flat = flat - dark
        radio = radio - dark
    valid = np.where(flat != 0)
    result = np.zeros(radio.shape, dtype=np.float32)
    result[valid] = radio[valid] / flat[valid]

    return result


def ramp_filter(width):
    """Get a 1D ramp filter for filtering sinogram rows."""
    base = np.arange(-width // 2, width // 2)

    return np.fft.fftshift(np.abs(base)) * 2.0 / width


@background
async def find_needle_tips(producer):
    """Get sample tips in images from *producer*."""
    tips = []
    coros = []

    async for image in producer:
        # start forces the coroutine to start immediately
        coros.append(run_in_executor(find_needle_tip, image))

    tips = [tip for tip in await asyncio.gather(*coros) if tip is not None]
    LOG.debug("Needle tips: %s", np.array(tips).tolist())

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


@background
async def find_sphere_centers_by_mass(producer, border_crossing_ok=True):
    """Get sphere centers in images from *producer* by computing their center of mass. The images
    must be absorption images. If *border_crossing_ok* is False skip images where sphere goes
    outside the field of view.
    """

    def _process_one(image):
        mask = segment_convex_object(image)
        mean_bg = image[mask == 0].mean()
        # Subtract mean of the background to correct for a global grey value offset
        tip = center_of_mass(image - mean_bg)
        if not border_crossing_ok and _touches_border(mask):
            LOG.debug(
                "Skipping border-crossing image with center of mass (x, y) = %s",
                tip[::-1],
            )
            tip = None

        return tip

    coros = []
    async for image in producer:
        coros.append(run_in_executor(_process_one, image))

    return [tip for tip in await asyncio.gather(*coros) if tip is not None]


@background
async def find_sphere_centers(producer, supersampling=1, correlation_threshold=None):
    """Get sphere centers in images from *producer*.  by finding the image with the largest portion
    of a sphere inside (the sphere may partially go out of the FOV) and correlate other images with
    the found one, from which relative shifts are computed and converted to absolute sphere centers.
    This is done by first computing the center of mass of the best image and then subtracting the
    respective shifts. Use *supersampling* for sub-pixel precision and filter out the centers for
    which the correlation coefficient computed by :func:`.compute_pearson_correlation_coefficient`
    is worse than *correlation_threshold*. The correlation coefficient is computed by shifting an
    image based on the shift found by correlation and computing the correlation coefficient of such
    shifted image with respect to the best one.
    """

    def _wrap(tips, axis):
        t = tips[:, axis]
        indices = np.where(t >= images[0].shape[axis])
        t[indices] = t[indices] - images[0].shape[axis]
        indices = np.where(t < 0)
        t[indices] = t[indices] + images[0].shape[axis]

    def _process_one(image):
        mask = segment_convex_object(image)
        return (image, mask, not _touches_border(mask))

    coros = []
    masks = []
    images = []
    found_completely_in_fov = False
    async for image in producer:
        coros.append(run_in_executor(_process_one, image))

    results = await asyncio.gather(*coros)

    for i, (image, mask, in_fov) in enumerate(results):
        images.append(image)
        masks.append(mask)
        if in_fov:
            a = image
            found_completely_in_fov = True
            LOG.debug("Sphere completely in FOV in image %d", i)

    if not found_completely_in_fov:
        nonzero = [np.count_nonzero(msk) for msk in masks]
        i = np.argmax(nonzero)
        LOG.debug("No sphere commpletely in FOV, largest portion in image %d", i)
        a = images[i]

    center_a = np.mean(np.where(segment_convex_object(a)), axis=1)
    shifts = np.array([correlate(a, b, supersampling=supersampling)[:2] for b in images])
    if correlation_threshold:
        r = np.empty(len(shifts))
        for i, (dy, dx) in enumerate(shifts):
            r[i] = await run_in_executor(
                compute_pearson_correlation_coefficient,
                a,
                images[i],
                int(np.round(dx)),
                int(np.round(dy)),
            )
        LOG.debug("Correlation coefficients: %s", r)
        shifts = shifts[np.where(r > correlation_threshold)]

    tips = center_a - shifts
    _wrap(tips, 0)
    _wrap(tips, 1)

    return tips


def segment_convex_object(image):
    """
    Extract convex object from *image* (e.g. needle or sphere). It doesn't matter if object is
    brigher or darker than the background (e.g. non flat corrected radiograph on input).
    """
    try:
        from skimage.filters import threshold_otsu
        from skimage.morphology import dilation, disk, convex_hull_image, label
    except ImportError as e:
        print("You need to install scikit-image in order to use this function")
        LOG.error(e)

    def _segment(threshold, greater=True):
        mask = np.zeros_like(image, dtype=np.int8)
        if greater:
            mask[image > threshold] = 1
        else:
            mask[image < threshold] = 1
        labels, num = label(mask, return_num=True)
        if not num:
            return None
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
    if mask is None:
        # Nothing found
        return None

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

    # Refine the segmentation of an object in image by setting the threshold to roughly FWTM of
    # the background standard deviation. Find the background by dilating the mask (in case some
    # object pixels are in the mask) by a small disk and then taking the inverse. sgn controls
    # whether the object is dark or bright (sgn = 1 for bright).
    indices = np.where(1 - dilation(mask, footprint=disk(20)))
    if not len(indices[0]):
        return None
    mean_bg = image[indices].mean()
    std_bg = image[indices].std()
    thr = mean_bg + sgn * 5 * std_bg
    if sgn == 1 and thr < thr_otsu or sgn == -1 and thr > thr_otsu:
        mask = _segment(thr, greater=sgn == 1)

    return mask


def _touches_border(mask):
    y, x = np.where(mask)

    return min(y) == 0 or max(y) == mask.shape[0] - 1 or min(x) == 0 or max(x) == mask.shape[1] - 1


def _find_peak_subpix(peak, image, supersampling=16):
    """Supersample vertical line at the *peak* (y, x) position by *supersampling* and look for the
    steepest gradient in the region (peak[0] - 1, peak[0] + 1) in the high resolution line.
    """
    from scipy.ndimage import gaussian_filter1d

    dy = 8
    y_start = max(peak[0] - dy, 0)
    line = image[y_start : min(peak[0] + dy, image.shape[0]), peak[1]]
    x = np.arange(len(line))
    x_hd = np.arange(0, len(line) - 1 + 1.0 / supersampling, 1.0 / supersampling)
    line_hd = np.interp(x_hd, x, line)
    # FWHM of the low resolution pixel
    sigma = supersampling / (2.0 * np.sqrt(2 * np.log(2)))
    blurred = gaussian_filter1d(line_hd, sigma)
    middle = len(x) * supersampling // 2
    g = np.abs(np.gradient(blurred))[middle - supersampling : middle + supersampling + 1]
    y = (np.argmax(g) + middle - supersampling) / supersampling + y_start

    return (y, peak[1])


def _get_boundary_coordinates(coordinates, max_val):
    """Return coordinates which reside on image edges."""
    return [coor for coor in coordinates if coor % max_val == 0]


def _is_corner_point(point, shape):
    """Test if the *point* lies in one of the image corners."""
    return (point[1] == 0 or point[1] == shape[1] - 1) and (
        point[0] == 0 or point[0] == shape[0] - 1
    )


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
        return [((p_1[0] + p_2[0]) / 2, p_1[1])]

    p_x = (p_1[1] + p_2[1]) / 2
    p_y = (p_1[0] + p_2[0]) / 2
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
    y_ind, x_ind = list(zip(*points))

    c_y = np.sum(y_ind) / len(points)
    c_x = np.sum(x_ind) / len(points)

    return c_y, c_x


def center_of_mass(frame):
    """Calculates the center of mass of the whole frame wheighted by value."""

    frm_shape = np.array(frame.shape)
    total = frame.sum()
    if total == 0:
        return np.array([-1, -1])
    else:
        y = (frame.sum(1) * np.arange(frm_shape[0])).sum() / total
        x = (frame.sum(0) * np.arange(frm_shape[1])).sum() / total
        return np.array([y, x])


def correlate(first, second, first_y=0, second_y=0, overlap_height=None, supersampling=1):
    """Correlate *first* and *second* image, use *supersampling* for sub-pixel precision. Crop first
    image vertically to (*first_y*, *first_y* + *overlap_height*) and second to (*second_y*,
    *second_y* + *overlap_height*).
    """
    from numpy.fft import fft2, ifft2, fftshift

    try:
        from skimage.filters import sobel
        from skimage.transform import resize
    except ImportError as e:
        print("You need to install scikit-image in order to use this function")
        LOG.error(e)

    height, width = first.shape
    hd_shape = (supersampling * height, supersampling * width)
    first = resize(first, hd_shape, order=1, mode="reflect")
    second = resize(second, hd_shape, order=1, mode="reflect")
    if supersampling > 1:
        ssh = supersampling // 2
        first = first[ssh:-ssh, ssh:-ssh]
        second = second[ssh:-ssh, ssh:-ssh]
    first_y = supersampling * first_y
    if overlap_height:
        overlap_height = supersampling * overlap_height
    else:
        overlap_height = second.shape[0]
    if second_y:
        second_y = supersampling * second_y
    else:
        second_y = second.shape[0] - overlap_height

    first_sobel = sobel(first)[first_y : first_y + overlap_height]
    second_sobel = sobel(second)[second_y : second_y + overlap_height]
    c = fftshift(ifft2(fft2(first_sobel) * np.conjugate(fft2(second_sobel))).real)
    dy, dx = np.unravel_index(c.argmax(), c.shape) - np.array(c.shape) / 2
    dy += second_y - first_y

    return (dy / supersampling, dx / supersampling, c)


def compute_pearson_correlation_coefficient(first, second, dx, dy):
    """Compute Pearson correlation coefficient. Image *second* is shifted by *dx* and *dy* pixels
    and the correlation is computed with respect to image *first*. Both images are cropped with
    respect to the *dx*, *dy* shift in order not to correlate regions overflowing over image edges.
    """
    first = first.copy()
    second = np.roll(np.roll(second.copy(), dy, axis=0), dx, axis=1)
    height, width = first.shape

    if abs(dx) >= width or abs(dy) >= height:
        r = 0
    else:
        # Both images must be cropped with respect to found dx and dy in order not to take garbage
        # into account.
        x_low = max(0, dx)
        x_high = dx if dx < 0 else None
        y_low = max(0, dy)
        y_high = dy if dy < 0 else None
        first = first[y_low:y_high, x_low:x_high]
        first -= first.mean()
        second = second.copy()[y_low:y_high, x_low:x_high]
        second -= second.mean()
        first_std = first.std()
        second_std = second.std()
        if not (first_std and second_std):
            r = 0
        else:
            r = np.mean(first * second) / (first_std * second_std)

    return r


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
    convolved = fftconvolve(first_projection, last_projection[::-1, :], mode="same")
    center = np.unravel_index(convolved.argmax(), convolved.shape)[1]

    return (width / 2 + center) / 2 * q.px


def filter_low_frequencies(data, fwhm=32.0):
    """Filter low frequencies in 1D *data*. *fwhm* is the FWHM of the gaussian used to filter out
    low frequencies in real space. The window is then computed as fft(1 - gauss).
    """
    mean = np.mean(data)
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    # We compute the gaussian in Fourier space, so convert sigma first
    f_sigma = 1.0 / (2 * np.pi * sigma)
    x = np.fft.fftfreq(len(data))
    fltr = 1 - np.exp(-(x**2) / (2 * f_sigma**2))

    return np.fft.ifft(np.fft.fft(data) * fltr).real + mean


def compute_frc(
    img1: ArrayLike,
    img2: ArrayLike,
    eps: float = 1e-12,
    apply_window: bool = True,
    window_alpha: float = 0.125,
    threshold_method: str = "1/7",
) -> Dict[str, Union[ArrayLike, float]]:
    """
    Compute Fourier Ring Correlation (FRC) between two images and estimate spatial resolution.

    FRC measures correlation between two images in Fourier space as a function of spatial
    frequency, providing a resolution estimate based on threshold crossing points.

    Algorithm:
        1. Apply Tukey window (reduces FFT edge artifacts), subtract DC component
        2. Compute 2D FFT of both images
        3. Calculate cross-correlation and power spectra
        4. Bin frequencies into concentric rings using radial distances
        5. Accumulate correlations per ring via bincount
        6. Compute FRC curve: normalized correlation per frequency ring
        7. Generate classical threshold curve (1/7 or half-bit method)
        8. Compute geometric bound (Miqueles et al., 2025) - ALWAYS computed

    :param img1: first input image (must have same shape as img2)
    :type img1: `concert.typing.ArrayLike`
    :param img2: second input image (must have same shape as img1)
    :type img2: `concert.typing.ArrayLike`
    :param eps: numerical stability constant to avoid division by zero (default: 1e-12)
    :type eps: float
    :param apply_window: apply Tukey window to reduce FFT artifacts (default: True)
    :type apply_window: bool
    :param window_alpha: Tukey window alpha (0=rectangular, 1=Hann; default: 0.125)
    :type window_alpha: float
    :param threshold_method: classical threshold method:
                             '1/7' - constant threshold at 1/7 ≈ 0.143
                             'half_bit' - information-theoretic threshold:
                                         (0.2071√n + 1.9102) / (1.2071√n + 0.9102)
    :type threshold_method: str
    :return: dictionary with keys:
             - 'frequencies': spatial frequency array (cycles/pixel)
             - 'frc': FRC curve (correlation per frequency bin)
             - 'classical_threshold': selected threshold curve
             - 'classical_crossing': frequency at classical threshold crossing (NaN if none)
             - 'classical_resolution': resolution = 1/classical_crossing (pixels, NaN if no crossing)
             - 'geometric_threshold': geometric lower bound (ALWAYS computed)
             - 'geometric_crossing': frequency at geometric bound crossing (NaN if none)
             - 'geometric_resolution': resolution = 1/geometric_crossing (pixels, NaN if no crossing)
    :rtype: Dict[str, Union[ArrayLike, float]]

    Notes:
        Resolution extraction uses linear interpolation between adjacent frequency bins for
        sub-bin precision. If no crossing found (FRC always above threshold), crossing
        frequencies and resolutions are NaN.

        Classical vs Geometric thresholds have DIFFERENT interpretations:

        **Classical** (1/7, half_bit): UPPER BOUND on resolution
            - Crossing indicates frequency where noise dominates signal
            - Use for resolution claims: "features ≥ X pixels are reliably resolved"

        **Geometric** (Miqueles et al., 2025): LOWER BOUND on expected correlation
            - Based on reverse Cauchy-Schwarz inequality
            - Quality assurance metric, NOT for resolution estimation
            - Crossing suggests data quality issues - investigate systematic errors

        Best practice: Report classical resolution with geometric_threshold as QA validation.

    References:
        - Van Heel, M. (1987). Similarity measures between images. Ultramicroscopy, 21(1), 95-100.
        - Nieuwenhuizen et al. (2013). Measuring image resolution in optical nanoscopy.
          Nature Methods, 10(6), 557-562. https://doi.org/10.1038/nmeth.2448
        - Van Heel, M., & Schatz, M. (2005). Fourier shell correlation threshold criteria.
          Journal of Structural Biology, 151(3), 250-262.
        - Miqueles, E. X., Tonin, Y. R., & Luke, R. D. (2025). A Novel Bound for Fourier
          Ring Correlation in Resolution Analysis. IEEE Transactions on Computational
          Imaging, 11, 1047-1058. https://doi.org/10.1109/TCI.2025.3593881
    """
    # Validate input shapes - FRC requires comparable Fourier spaces
    if img1.shape != img2.shape:
        raise ValueError(
            f"incompatible image shapes, image1 shape: {img1.shape}, image2 shape: {img2.shape}"
        )

    # Convert to array type (NumPy or CuPy depending on availability)
    img1 = xp.asarray(img1, dtype=xp.float64)
    img2 = xp.asarray(img2, dtype=xp.float64)
    height, width = img1.shape

    # Apply Tukey window to suppress FFT edge artifacts
    if apply_window:
        try:
            tukey_1d = xp_windows.tukey(max(height, width), alpha=window_alpha)
            window_2d = tukey_1d[:height, None] * tukey_1d[None, :width]
            img1 = img1 * window_2d
            img2 = img2 * window_2d
        except Exception:
            LOG.warning("Tukey window not available, proceeding without windowing")

    # Remove DC component - prevents low-frequency dominance in FRC
    img1 = img1 - img1.mean()
    img2 = img2 - img2.mean()

    # Compute 2D FFT for both images
    freq1: ArrayLike = xft.fft2(img1)
    freq2: ArrayLike = xft.fft2(img2)

    # Cross-correlation spectrum: measures agreement in amplitude and phase
    corr: ArrayLike = (freq1 * xp.conj(freq2)).real

    # Power spectra: normalize for differences in signal strength
    pow_spc1: ArrayLike = xp.abs(freq1) ** 2
    pow_spc2: ArrayLike = xp.abs(freq2) ** 2

    # Create frequency coordinate grid (units: cycles/pixel)
    # fftfreq returns frequencies from -Nyquist to +Nyquist
    freq_y, freq_x = xp.meshgrid(xft.fftfreq(height), xft.fftfreq(width), indexing="ij")

    # Compute radial distance from origin for each frequency pixel using Pythagorean formula.
    # This converts 2D frequency coordinates to scalar spatial frequency magnitudes
    radial_distances: ArrayLike = xp.sqrt(freq_x**2 + freq_y**2)

    # Number of frequency bins determined by Nyquist limit
    # We can only resolve up to min(height, width) // 2 independent frequency rings
    num_freq_bins: int = min(height, width) // 2

    # Define bin edges as equally-spaced radii from 0 to maximum frequency
    ring_radii = xp.linspace(0, radial_distances.max(), num_freq_bins + 1)

    # Assign each frequency pixel to a frequency ring (bin) based on its radial distance
    # digitize() finds which interval [ring_radii[i], ring_radii[i+1]) each distance falls into
    # Returns integer array where value k means "this pixel belongs to ring k"
    # Subtract 1 because digitize uses 1-based indexing but we need 0-based for bincount
    bin_idx = xp.digitize(radial_distances.ravel(), bins=ring_radii) - 1

    # Clip to valid range handles floating-point precision edge cases
    bin_idx = xp.clip(bin_idx, 0, num_freq_bins - 1)

    # At this point, bin_idx is a flattened array (same length as total pixels)
    # where each element indicates which frequency ring that pixel contributes to

    # Accumulate cross-correlations per frequency ring
    # bincount(indices, weights) sums all weights that share the same index value
    frc_num: ArrayLike = xp.bincount(bin_idx, weights=corr.ravel(), minlength=num_freq_bins)

    # Similarly accumulate power spectra for normalization
    # These represent total signal strength per frequency ring for each image
    frc_denom1: ArrayLike = xp.bincount(bin_idx, weights=pow_spc1.ravel(), minlength=num_freq_bins)
    frc_denom2: ArrayLike = xp.bincount(bin_idx, weights=pow_spc2.ravel(), minlength=num_freq_bins)

    # Count pixels per ring (no weights) - used for reliability masking and thresholds
    counts: ArrayLike = xp.bincount(bin_idx, minlength=num_freq_bins)

    # Compute FRC: normalized correlation per frequency ring
    # Formula: FRC(k) = Σ(corr_k) / √(Σ(pow1_k)  Σ(pow2_k))
    # Add eps to denominator to prevent division by zero for empty/high-frequency rings
    frc: ArrayLike = frc_num / (xp.sqrt(frc_denom1 * frc_denom2) + eps)

    # Mask unreliable bins - rings with <10 pixels have poor statistical significance
    frc = xp.where(counts > 10, frc, xp.nan)

    # Compute representative frequency for each bin (bin center, not edge)
    frequencies: ArrayLike = 0.5 * (ring_radii[:-1] + ring_radii[1:])

    # Generate classical threshold curve based on selected method
    if threshold_method == "1/7":
        # Classic fixed threshold at 1/7 ≈ 0.143
        threshold_curve = xp.ones_like(frequencies) * (1.0 / 7.0)
    elif threshold_method == "half_bit":
        # Information-theoretic threshold based on pixel counts per ring
        # Formula: (snr√n + (factor+1)) / ((snr+1)√n + factor)
        # where snr = 0.5√2 - 0.5 ≈ 0.2071, factor = √snr  2 ≈ 0.9102
        nr_rt = xp.sqrt(counts)
        snr_half_set = 0.5 * xp.sqrt(2) - 0.5
        factor = xp.sqrt(snr_half_set) * 2
        threshold_curve = (snr_half_set * nr_rt + (factor + 1)) / (
            (snr_half_set + 1) * nr_rt + factor
        )
    else:
        raise ValueError(
            f"Unknown threshold_method '{threshold_method}'. " f"Valid options: '1/7', 'half_bit'"
        )

    # Compute geometric threshold.
    # This is the lower bound from Miqueles et al. (2025) based on reverse Cauchy-Schwarz
    fft1_shifted = xft.fftshift(freq1)
    fft2_shifted = xft.fftshift(freq2)
    mag1 = xp.abs(fft1_shifted)
    mag2 = xp.abs(fft2_shifted)
    M_per_ring = xp.bincount(
        bin_idx,
        weights=xp.maximum(mag1.ravel(), mag2.ravel()),
        minlength=num_freq_bins,
    )
    M_per_ring = M_per_ring / xp.maximum(counts, 1)  # Average, not sum
    geometric_threshold = 2.0 * xp.sqrt(M_per_ring) / (1.0 + M_per_ring)
    geometric_threshold = xp.where(counts > 10, geometric_threshold, xp.nan)

    # Extract spatial resolution from classical threshold crossing point
    classical_crossing, classical_resolution = _extract_resolution(
        frequencies, frc, threshold_curve, "classical"
    )

    # Extract spatial resolution from geometric threshold crossing point
    geometric_crossing, geometric_resolution = _extract_resolution(
        frequencies, frc, geometric_threshold, "geometric"
    )

    return {
        "frequencies": frequencies,
        "frc": frc,
        "classical_threshold": threshold_curve,
        "classical_crossing": classical_crossing,
        "classical_resolution": classical_resolution,
        "geometric_threshold": geometric_threshold,
        "geometric_crossing": geometric_crossing,
        "geometric_resolution": geometric_resolution,
    }


def _extract_resolution(
    frequencies: ArrayLike,
    frc_curve: ArrayLike,
    threshold_curve: ArrayLike,
    crossing_type: str = "classical",
) -> Tuple[float, float]:
    """
    Extract resolution from FRC curve threshold crossing using linear interpolation.

    :param frequencies: spatial frequency array (cycles/pixel)
    :param frc_curve: FRC correlation values
    :param threshold_curve: threshold values to compare against
    :param crossing_type: 'classical' or 'geometric' (for logging purposes)
    :return: (crossing_frequency, resolution) tuple, both NaN if no valid crossing
    """
    # Find all frequency bins where FRC drops below threshold
    crossing_mask: ArrayLike = frc_curve < threshold_curve

    # Search for first valid crossing where both bins have valid FRC values
    crossing_frequency = float("nan")
    resolution = float("nan")

    for first_idx in range(1, len(frequencies)):
        if (
            crossing_mask[first_idx]
            and not xp.isnan(frc_curve[first_idx])
            and not xp.isnan(frc_curve[first_idx - 1])
        ):
            # Valid crossing found - use linear interpolation for sub-bin precision

            # Extract values from adjacent bins for interpolation
            f1 = float(frequencies[first_idx - 1])
            f2 = float(frequencies[first_idx])
            c1 = float(frc_curve[first_idx - 1])
            c2 = float(frc_curve[first_idx])
            t1 = float(threshold_curve[first_idx - 1])
            t2 = float(threshold_curve[first_idx])

            # Linear interpolation: solve for frequency where FRC = threshold
            # Formula: f_cross = f1 + (t - c1) * (f2 - f1) / ((c2 - c1) - (t2 - t1))
            denom = (c2 - c1) - (t2 - t1)
            if abs(denom) < 1e-10:
                # Near-parallel curves - fall back to midpoint
                LOG.debug(
                    "Near-parallel FRC and %s threshold curves - using midpoint interpolation",
                    crossing_type,
                )
                crossing_frequency = (f1 + f2) / 2
            else:
                crossing_frequency = f1 - (c1 - t1) * (f2 - f1) / denom

            # Convert crossing frequency to spatial resolution
            if xp.isnan(crossing_frequency) or crossing_frequency <= 0:
                LOG.debug(
                    "Invalid %s crossing frequency %.4f - setting resolution to NaN",
                    crossing_type,
                    crossing_frequency,
                )
                resolution = float("nan")
            else:
                resolution = 1.0 / crossing_frequency
                LOG.debug(
                    "%s resolution extracted: %.4f cycles/pixel → %.2f pixels",
                    crossing_type.capitalize(),
                    crossing_frequency,
                    resolution,
                )
            break

    if xp.isnan(crossing_frequency):
        LOG.debug("No valid %s threshold crossing found", crossing_type)

    return crossing_frequency, resolution
