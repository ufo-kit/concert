"""
frc.py
-----
Implements a device server to execute Fourier Ring Correlation during acquisition.
"""
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union
import json
import math
import os
import datetime
import numpy as np
from tango import DebugIt
from tango.server import attribute, command, AttrWriteType
try:
    import torch
    import torch.nn.functional as F
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    _device = torch.device("cpu")
from scipy.ndimage import uniform_filter
from scipy.signal.windows import tukey
from concert.ext.tangoservers.base import TangoRemoteProcessing, RemoteWalkerMixin
from concert.typing import ArrayLike
from concert.imageprocessing import flat_correct
from concert.storage import RemoteDirectoryWalker
from concert.helpers import PerformanceTracker
from concert.typing import ArrayLike


LOG = logging.getLogger(__name__)


def select_frc_region(
    projection: ArrayLike,
    crop_height: int,
    padding_y: int,
    padding_x: int,
) -> Tuple[int, int, str]:
    """
    Select informative vertical region from projection for FRC computation.

    Identifies the y-position with maximum information content (variance) and returns
    crop boundaries. The full width is retained to ensure the sample remains in the
    field of view during tomographic rotation.

    :param projection: input projection image
    :type projection: `concert.typing.ArrayLike`
    :param crop_height: height of cropped region in pixels (default: 512)
    :type crop_height: int
    :param padding_y: vertical padding in pixels to exclude from top/bottom edges
                      (default: 200). Variance computation ignores these regions.
    :type padding_y: int
    :param padding_x: horizontal padding in pixels to exclude from left/right edges
                      (default: 200). Variance computation ignores these regions.
    :type padding_x: int
    :return: (y_start, y_end, crop_method) crop boundaries and method used
             crop_method is "variance" (peak detected) or "center" (uniform variance)
    :rtype: Tuple[int, int, str]

    Notes:
        - Uses local variance to identify information-rich regions (edges, structures)
        - Variance computed efficiently via: Var(X) = E[X²] - E[X]²
        - If variance map is uniform (no clear maximum), defaults to center crop
        - Crop boundaries are clamped to image dimensions
        - Full width is always retained to accommodate sample rotation
    """
    proj = torch.as_tensor(projection, dtype=torch.float64, device=_device)
    height, width = proj.shape
    assert padding_y > 0 and padding_y * 2 < height
    assert padding_x > 0 and padding_x * 2 < width
    variance_window = max(1, crop_height // 4)

    # Validate crop_height fits in padded region
    inner_height = height - (padding_y * 2)
    if crop_height > inner_height:
        raise ValueError(
            f"crop height ({crop_height}px) exceeds available region ({inner_height}px)")

    # Apply padding: exclude edge regions from variance computation
    # This prevents empty air/artifacts from overwhelming the detection
    y_inner_start = padding_y
    y_inner_end = height - padding_y
    x_inner_start = padding_x
    x_inner_end = width - padding_x

    # Extract inner region for variance analysis
    proj_inner = proj[y_inner_start:y_inner_end, x_inner_start:x_inner_end]

    # Use scipy.ndimage.uniform_filter (NumPy arrays)
    proj_inner_cpu = proj_inner.cpu().numpy()
    mean = uniform_filter(proj_inner_cpu, size=variance_window)
    mean_sq = uniform_filter(proj_inner_cpu**2, size=variance_window)

    # Convert back to torch tensor
    mean = torch.as_tensor(mean, dtype=torch.float64, device=_device)
    mean_sq = torch.as_tensor(mean_sq, dtype=torch.float64, device=_device)

    # Variance = E[X²] - E[X]²
    variance_2d = mean_sq - mean**2

    # Collapse variance map horizontally (sum across all columns)
    # This gives us a 1D profile showing information content vs. y-position
    variance_1d = variance_2d.sum(dim=1)

    # Check for uniform variance (all values nearly equal)
    # If variance range is very small, default to center crop
    variance_range = float(variance_1d.max() - variance_1d.min())
    if variance_range < 1e-6:
        # Uniform variance - default to center
        center_y = height // 2
        max_variance = float(variance_1d[center_y].item())
        LOG.debug(
            "Uniform variance detected (range=%.2e), using center crop at y=%d",
            variance_range,
            center_y,
        )
        crop_method = "center"
    else:
        # Find y-position with maximum total variance
        # This is the "center of mass" of information in the vertical direction
        max_y_flat = torch.argmax(variance_1d)
        center_y = int(max_y_flat.item())
        max_variance = float(variance_1d[max_y_flat].item())
        crop_method = "variance"

    # Compute crop boundaries centered at max variance position
    y_start = center_y - crop_height // 2
    y_end = y_start + crop_height

    # Clamp to image boundaries
    if y_start < 0:
        y_start = 0
        y_end = min(crop_height, height)

    if y_end > height:
        y_end = height
        y_start = max(0, height - crop_height)

    LOG.debug(
        "Selected FRC region: y=[%d:%d] (height=%d, center_y=%d, max_variance=%.2f, method=%s)",
        y_start,
        y_end,
        y_end - y_start,
        center_y,
        max_variance,
        crop_method,
    )

    return y_start, y_end, crop_method


def prepare_frc_state(height: int, width: int) -> Dict[str, Union[int, torch.Tensor]]:
    """
    Precompute frequency bins and bin assignments for FRC computation.

    This function computes shape-dependent data structures that can be reused
    across multiple FRC computations for images of the same dimensions.

    :param height: image height in pixels
    :type height: int
    :param width: image width in pixels
    :type width: int
    :return: dictionary containing:
             - 'height': image height
             - 'width': image width
             - 'num_freq_bins': number of frequency rings
             - 'bin_idx': flattened bin assignments for each pixel
             - 'counts': number of pixels per frequency ring
             - 'frequencies': representative frequency for each ring
    :rtype: Dict[str, Union[int, torch.Tensor]]
    :raises ValueError: if height or width < 2
    """
    # Validate dimensions
    assert height > 2 and width > 2
    # Create frequency coordinate grid
    freq_y_vals = torch.fft.fftfreq(height, device=_device)
    freq_x_vals = torch.fft.fftfreq(width, device=_device)
    freq_y, freq_x = torch.meshgrid(freq_y_vals, freq_x_vals, indexing="ij")

    # Compute radial distances
    radial_distances = torch.sqrt(freq_x**2 + freq_y**2)

    # Number of frequency bins determined by Nyquist limit
    num_freq_bins: int = min(height, width) // 2

    # Define ring radii (bin edges)
    ring_radii = torch.linspace(0, radial_distances.max(), num_freq_bins + 1, device=_device)

    # Assign each frequency pixel to a bin
    bin_idx = torch.bucketize(radial_distances.ravel(), ring_radii, right=True) - 1
    bin_idx = torch.clamp(bin_idx, 0, num_freq_bins - 1)

    # Count pixels per ring
    counts = torch.bincount(bin_idx, minlength=num_freq_bins)

    # Compute representative frequencies (bin centers)
    frequencies = 0.5 * (ring_radii[:-1] + ring_radii[1:])

    return {
        "height": height,
        "width": width,
        "num_freq_bins": num_freq_bins,
        "bin_idx": bin_idx,
        "counts": counts,
        "frequencies": frequencies,
    }


def _frc_core(
    img1: torch.Tensor,
    img2: torch.Tensor,
    dark: Optional[torch.Tensor],
    flat: Optional[torch.Tensor],
    window_2d: torch.Tensor,
    bin_idx: torch.Tensor,
    counts: torch.Tensor,
    frequencies: torch.Tensor,
    num_freq_bins: int,
    eps: float = 1e-12,
    threshold_method: str = "half_bit") -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Core FRC computation with flat-field correction and windowing.

    Performs flat-field correction on input images, applies apodization window,
    computes FFT-based cross-correlation and power spectra, accumulates correlations
    per frequency ring using precomputed bin assignments, and generates threshold curves.

    :param img1: first image tensor
    :param img2: second image tensor
    :param dark: dark field reference
    :param flat: flat field reference
    :param window_2d: 2D edge smoothing Tukey window
    :param bin_idx: frequency bin assignments for each pixel
    :param counts: number of pixels per frequency ring
    :param frequencies: representative frequencies for each ring
    :param num_freq_bins: total number of frequency bins
    :param eps: small constant for numerical stability
    :param threshold_method: threshold method ('1/7' or 'half_bit')
    :return: tuple of (frc_curve, threshold_curve, geometric_threshold)
    """
    # Flat-field Correct
    if dark is not None and flat is not None:
        dark = torch.as_tensor(dark.copy(), dtype=torch.float64, device=_device)
        flat = torch.as_tensor(flat.copy(), dtype=torch.float64, device=_device)
        flat -= dark
        img1 = torch.where(
            flat != 0,
            torch.log((img1 - dark) / flat),
            torch.tensor(0.0, dtype=torch.float64, device=_device),
        )
        img2 = torch.where(
            flat != 0,
            torch.log((img2 - dark) / flat),
            torch.tensor(0.0, dtype=torch.float64, device=_device),
        )

    # Apply Tukey window
    img1 *= window_2d
    img2 *= window_2d
    
    # Compute cross correlation spectrum and power spectra
    freq1 = torch.fft.fft2(img1 - img1.mean())
    freq2 = torch.fft.fft2(img2 - img2.mean())
    corr = (freq1 * torch.conj(freq2)).real
    pow_spc1 = torch.abs(freq1) ** 2
    pow_spc2 = torch.abs(freq2) ** 2

    # Accumulate cross-correlations per frequency ring using precomputed bin assignments
    frc_num = torch.bincount(bin_idx, weights=corr.ravel(), minlength=num_freq_bins)

    # Accumulate power spectra for normalization
    frc_denom1 = torch.bincount(bin_idx, weights=pow_spc1.ravel(), minlength=num_freq_bins)
    frc_denom2 = torch.bincount(bin_idx, weights=pow_spc2.ravel(), minlength=num_freq_bins)

    # Compute FRC: normalized correlation per frequency ring and mask unreliable bins
    # Rings with <10 pixels have poor statistical significance.
    frc = frc_num / (torch.sqrt(frc_denom1 * frc_denom2) + eps)
    frc = torch.where(
        counts > 10, frc, torch.tensor(float("nan"), device=_device)
    )

    # Generate classical threshold curve based on selected method
    if threshold_method == "1/7":
        # Classic fixed threshold at 1/7 ≈ 0.143
        threshold_curve = torch.ones_like(frequencies) * (1.0 / 7.0)
    elif threshold_method == "half_bit":
        # Information-theoretic threshold based on pixel counts per ring
        # Formula: (snr√n + (factor+1)) / ((snr+1)√n + factor)
        # where snr = 0.5√2 - 0.5 ≈ 0.2071, factor = √snr  2 ≈ 0.9102
        nr_rt = torch.sqrt(counts)
        snr_half_set = 0.5 * torch.sqrt(torch.tensor(2.0, device=_device)) - 0.5
        factor = torch.sqrt(snr_half_set) * 2
        threshold_curve = (snr_half_set * nr_rt + (factor + 1)) / (
            (snr_half_set + 1) * nr_rt + factor
        )
    else:
        raise ValueError(
            f"Unknown threshold_method '{threshold_method}'. "
            f"Valid options: '1/7', 'half_bit'"
        )

    # Compute geometric threshold.
    fft1_shifted = torch.fft.fftshift(freq1)
    fft2_shifted = torch.fft.fftshift(freq2)
    mag1 = torch.abs(fft1_shifted)
    mag2 = torch.abs(fft2_shifted)
    M_per_ring = torch.bincount(
        bin_idx,
        weights=torch.maximum(mag1.ravel(), mag2.ravel()),
        minlength=num_freq_bins,
    )
    M_per_ring = M_per_ring / torch.maximum(counts, torch.ones_like(counts))
    geometric_threshold = 2.0 * torch.sqrt(M_per_ring) / (1.0 + M_per_ring)
    geometric_threshold = torch.where(
        counts > 10, geometric_threshold, torch.tensor(float("nan"), device=_device)
    )
    return frc, threshold_curve, geometric_threshold


def _extract_resolution(
    frequencies: torch.Tensor,
    frc_curve: torch.Tensor,
    threshold_curve: torch.Tensor,
    crossing_type: str = "classical",
) -> Tuple[float, float]:
    """
    Extract resolution from FRC curve threshold crossing using vectorized operations.

    Fully vectorized implementation replaces Python loop with tensor operations.
    Finds first frequency where FRC drops below threshold, using linear interpolation.

    :param frequencies: spatial frequency array (cycles/pixel)
    :type frequencies: torch.Tensor
    :param frc_curve: FRC correlation values
    :type frc_curve: torch.Tensor
    :param threshold_curve: threshold values to compare against
    :type threshold_curve: torch.Tensor
    :param crossing_type: 'classical' or 'geometric' (for logging purposes)
    :type crossing_type: str
    :return: (crossing_frequency, resolution) tuple, both NaN if no valid crossing
    :rtype: Tuple[float, float]
    """
    # Find all frequency bins where FRC drops below threshold
    crossing_mask = frc_curve < threshold_curve
    valid_mask = ~torch.isnan(frc_curve)

    # Find all indices where crossing_mask[i] is True and both i and i-1 are valid
    valid_crossings = crossing_mask[1:] & valid_mask[1:] & valid_mask[:-1]

    crossings = torch.where(valid_crossings)[0]

    if len(crossings) == 0:
        LOG.debug("No valid %s threshold crossing found", crossing_type)
        return float("nan"), float("nan")

    # Get first crossing index (add 1 because valid_crossings starts from index 1)
    first_idx = int(crossings[0]) + 1

    # Extract values from adjacent bins for interpolation
    f1 = frequencies[first_idx - 1]
    f2 = frequencies[first_idx]
    c1 = frc_curve[first_idx - 1]
    c2 = frc_curve[first_idx]
    t1 = threshold_curve[first_idx - 1]
    t2 = threshold_curve[first_idx]

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
    if torch.isnan(crossing_frequency) or crossing_frequency <= 0:
        LOG.debug(
            "Invalid %s crossing frequency %.4f - setting resolution to NaN",
            crossing_type, float(crossing_frequency)
        )
        return float("nan"), float("nan")
    resolution = 1.0 / crossing_frequency
    LOG.debug(
        "%s resolution: %.4f cycles/px → %.4f px",
        crossing_type.capitalize(), float(crossing_frequency), float(resolution)
    )
    return float(crossing_frequency), float(resolution)


def compute_frc(
    img1: ArrayLike,
    img2: ArrayLike,
    dark: Optional[ArrayLike],
    flat: Optional[ArrayLike],
    frc_state: Dict[str, Union[int, torch.Tensor]],
    eps: float = 1e-12,
    window_alpha: float = 0.125,
    threshold_method: str = "half_bit",
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
        8. Compute geometric bound (Miqueles et al., 2025)

    :param img1: first input image
    :type img1: `concert.typing.ArrayLike`
    :param img2: second input image
    :type img2: `concert.typing.ArrayLike`
    :param dark: dark field
    :type dark: `concert.typing.ArrayLike`
    :param flat: flat field
    :type flat: `concert.typing.ArrayLike`
    :param frc_state: precomputed frequency bins from prepare_frequency_bins()
    :type frc_state: Dict[str, Union[int, torch.Tensor]]
    :param eps: numerical stability constant to avoid division by zero (default: 1e-12)
    :type eps: float
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
    assert img1.shape == img2.shape
    # Convert to torch tensors on appropriate device
    img1 = torch.as_tensor(img1.copy(), dtype=torch.float64, device=_device)
    img2 = torch.as_tensor(img2.copy(), dtype=torch.float64, device=_device)
    height, width = frc_state["height"], frc_state["width"]
    
    # Apply Tukey window to suppress FFT edge artifacts
    tukey_1d = torch.as_tensor(
        tukey(max(height, width), alpha=window_alpha), dtype=torch.float64, device=_device)
    window_2d = tukey_1d[:height, None] * tukey_1d[None, :width]

    # Use precomputed frequency bins
    bin_idx = frc_state["bin_idx"]
    counts = frc_state["counts"]
    frequencies = frc_state["frequencies"]
    num_freq_bins = frc_state["num_freq_bins"]

    frc, threshold_curve, geometric_threshold = _frc_core(
        img1, img2, dark, flat, window_2d,
        frc_state["bin_idx"], frc_state["counts"], frc_state["frequencies"],
        frc_state["num_freq_bins"], eps, threshold_method
    )

    # Extract spatial resolution from classical threshold crossing point
    classical_crossing, classical_resolution = _extract_resolution(
        frequencies, frc, threshold_curve, "classical"
    )

    # Extract spatial resolution from geometric threshold crossing point
    geometric_crossing, geometric_resolution = _extract_resolution(
        frequencies, frc, geometric_threshold, "geometric"
    )

    # Convert all torch tensors back to numpy for API compatibility
    return {
        "frequencies": frequencies.cpu().numpy(),
        "frc": frc.cpu().numpy(),
        "classical_threshold": threshold_curve.cpu().numpy(),
        "classical_crossing": float(classical_crossing),
        "classical_resolution": float(classical_resolution),
        "geometric_threshold": geometric_threshold.cpu().numpy(),
        "geometric_crossing": float(geometric_crossing),
        "geometric_resolution": float(geometric_resolution),
    }


class TangoFourierRingCorrelation(TangoRemoteProcessing, RemoteWalkerMixin):
    """
    Implements Tango device server to compute Fourier Ring Correlation for resolution estimation.
    """

    attr_acq = attribute(
        label="Meta attribute for acquisition",
        dtype=(int,),
        max_dim_x=3,
        access=AttrWriteType.READ_WRITE,
        fget="get_attr_acq",
        fset="set_attr_acq",
        doc="encapsulates acquisition meta information i.e., #darks, #flats, #radios",
    )

    proj_offset = attribute(
        label="Projection offset",
        dtype=int,
        access=AttrWriteType.READ_WRITE,
        fget="get_proj_offset",
        fset="set_proj_offset",
        doc="offset in projections to compute FRC",
    )

    resolution_threshold = attribute(
        label="FRC resolution threshold",
        dtype=str,
        access=AttrWriteType.READ_WRITE,
        fget="get_resolution_threshold",
        fset="set_resolution_threshold",
        doc="Threshold for resolution determination: '1/7', 'half_bit'",
    )

    fluctuation_threshold = attribute(
        label="FRC fluctuation threshold percentage",
        dtype=float,
        access=AttrWriteType.READ_WRITE,
        fget="get_fluctuation_threshold",
        fset="set_fluctuation_threshold",
        doc="Percentage deviation from baseline to flag as fluctuation"
        "(default: 15.0, valid range: 5.0-100.0)",
    )

    crop_height = attribute(
        label="Crop height",
        dtype=int,
        access=AttrWriteType.READ_WRITE,
        fget="get_crop_height",
        fset="set_crop_height",
        doc="Target crop height in pixels for FRC region selection (default: 512)",
    )

    padding_y = attribute(
        label="Vertical padding",
        dtype=int,
        access=AttrWriteType.READ_WRITE,
        fget="get_padding_y",
        fset="set_padding_y",
        doc="Vertical padding in pixels to exclude from top/bottom edges (default: 400)",
    )

    padding_x = attribute(
        label="Horizontal padding",
        dtype=int,
        access=AttrWriteType.READ_WRITE,
        fget="get_padding_x",
        fset="set_padding_x",
        doc="Horizontal padding in pixels to exclude from left/right edges (default: 400)",
    )

    y_start = attribute(
        label="Crop Y start",
        dtype=int,
        access=AttrWriteType.READ_WRITE,
        fget="get_y_start",
        fset="set_y_start",
        doc="Starting Y coordinate for FRC region selection (default: 0)",
    )

    y_end = attribute(
        label="Crop Y end",
        dtype=int,
        access=AttrWriteType.READ_WRITE,
        fget="get_y_end",
        fset="set_y_end",
        doc="Ending Y coordinate for FRC region selection (default: 0, determined from data)",
    )

    _walker: Optional[RemoteDirectoryWalker]
    _frc_state: Optional[Dict[str, Union[int, torch.Tensor]]]
    _y_start: int
    _y_end: int

    async def init_device(self) -> None:
        await super().init_device()
        self._resolution_threshold = "half_bit"
        self._proj_offset = 1
        self._fluctuation_threshold = 15.0
        self._crop_height = 512
        self._padding_y = 200
        self._padding_x = 200
        self._y_start = 0
        self._y_end = 0
        self._frc_state = None
        self._walker = None
        self.info_stream(
            "%s initialized device with %s, state: %s",
            self.__class__.__name__,
            "CUDA" if torch.cuda.is_available() else "CPU",
            self.get_state(),
        )

    def get_attr_acq(self) -> ArrayLike:
        return self._attr_acq

    def set_attr_acq(self, aa: ArrayLike) -> None:
        self._attr_acq = aa
        self.info_stream(
            "%s: acquisition attributes set to: %s",
            self.__class__.__name__,
            str(self._attr_acq),
        )

    def get_proj_offset(self) -> int:
        return self._proj_offset

    def set_proj_offset(self, offset: int) -> None:
        self._proj_offset = offset
        self.info_stream(
            "%s: proj_offset set to: %s",
            self.__class__.__name__,
            str(self._proj_offset),
        )

    def get_resolution_threshold(self) -> str:
        return self._resolution_threshold

    def set_resolution_threshold(self, threshold: str) -> None:
        self._resolution_threshold = threshold
        self.info_stream(
            "%s: resolution_threshold set to: %s",
            self.__class__.__name__,
            str(self._resolution_threshold),
        )

    def get_fluctuation_threshold(self) -> float:
        return self._fluctuation_threshold

    def set_fluctuation_threshold(self, threshold: float) -> None:
        if threshold < 5.0 or threshold > 100.0:
            raise ValueError(
                f"Fluctuation threshold must be between 5.0 and 100.0 percent, " f"got {threshold}"
            )
        self._fluctuation_threshold = threshold
        self.info_stream(
            "%s: fluctuation_threshold set to: %s",
            self.__class__.__name__,
            f"{self._fluctuation_threshold}%",
        )

    def get_crop_height(self) -> int:
        return self._crop_height

    def set_crop_height(self, height: int) -> None:
        self._crop_height = height
        self.info_stream(
            "%s: crop_height set to: %s",
            self.__class__.__name__,
            str(self._crop_height),
        )

    def get_padding_y(self) -> int:
        return self._padding_y

    def set_padding_y(self, padding: int) -> None:
        if padding < 0:
            raise ValueError(f"Padding Y must be non-negative, got {padding}")
        self._padding_y = padding
        self.info_stream(
            "%s: padding_y set to: %s",
            self.__class__.__name__,
            str(self._padding_y),
        )

    def get_padding_x(self) -> int:
        return self._padding_x

    def set_padding_x(self, padding: int) -> None:
        if padding < 0:
            raise ValueError(f"Padding X must be non-negative, got {padding}")
        self._padding_x = padding
        self.info_stream(
            "%s: padding_x set to: %s",
            self.__class__.__name__,
            str(self._padding_x),
        )

    def get_y_start(self) -> int:
        return self._y_start
    
    def set_y_start(self, y_start: int) -> None:
        if y_start < 0:
            raise ValueError(f"Crop Y start must be non-negative, got {y_start}")
        self._y_start = y_start
        self.info_stream(
            "%s: y_start set to: %s",
            self.__class__.__name__,
            str(self._y_start),
        )

    def get_y_end(self) -> int:
        return self._y_end

    def set_y_end(self, y_end: int) -> None:
        if y_end < 0:
            raise ValueError(f"Crop Y end must be non-negative, got {y_end}")
        self._y_end = y_end
        self.info_stream(
            "%s: y_end set to: %s",
            self.__class__.__name__,
            str(self._y_end),
        )

    def _compute_centered_crop(self, y_start: int, y_end: int) -> Tuple[int, int]:
        """
        Compute centered crop boundaries within configured [y_start, y_end] interval.
        
        :param y_start: configured start coordinate
        :param y_end: configured end coordinate
        :return: tuple of (crop_y_start, crop_y_end) centered within interval
        """
        center_y = (y_start + y_end) // 2
        half_crop = self._crop_height // 2
        
        crop_start = center_y - half_crop
        crop_end = crop_start + self._crop_height
        
        # Clamp to stay within configured bounds
        if crop_start < y_start:
            crop_start = y_start
            crop_end = crop_start + self._crop_height
        
        if crop_end > y_end:
            crop_end = y_end
            crop_start = crop_end - self._crop_height
        
        return crop_start, crop_end

    @staticmethod
    def _result_to_json_dict(
        result: Dict[str, Union[ArrayLike, float]],
        proj_i: int,
        proj_j: int,
    ) -> Dict[str, Any]:
        """
        Convert compute_frc() result dict to JSON-serializable dictionary.

        :param result: dictionary returned by compute_frc()
        :param proj_i: first projection index
        :param proj_j: second projection index (proj_i + offset)
        :return: JSON-serializable dictionary
        """

        def _sanitize_for_json(value: float) -> Optional[float]:
            """Convert NaN/Inf to None for JSON compatibility."""
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                return None
            return value

        return {
            "projection_i": proj_i,
            "projection_j": proj_j,
            "classical_resolution": _sanitize_for_json(float(result["classical_resolution"])),
            "geometric_resolution": _sanitize_for_json(float(result["geometric_resolution"])),
        }

    @staticmethod
    def _compute_outlier_aware_statistics(
        values: List[Optional[float]],
        projection_indices: List[int],
        threshold_percent: float = 15.0,
    ) -> Dict[str, Any]:
        """
        Compute outlier-aware statistics for a resolution metric.

        Identifies projections where resolution deviates significantly from the baseline
        (median) value. Useful for detecting sparse fluctuations in otherwise stable
        tomographic acquisitions.

        :param values: list of resolution values (may contain None for NaN/invalid)
        :param projection_indices: corresponding projection_i indices for each value
        :param threshold_percent: percentage deviation from baseline to flag as fluctuation
        :return: dictionary with baseline, counts, and fluctuation details
        """
        # Filter out None/NaN values
        valid_pairs = [(v, i) for v, i in zip(values, projection_indices) if v is not None]

        if not valid_pairs:
            return {
                "baseline_median": None,
                "num_measurements": len(values),
                "num_valid": 0,
                "num_stable": 0,
                "num_fluctuations": 0,
                "fluctuation_threshold_percent": threshold_percent,
                "fluctuation_indices": [],
                "fluctuation_details": [],
            }

        # Extract just the values for median calculation
        valid_values = [v for v, i in valid_pairs]

        # Compute baseline (median)
        baseline_median = float(np.median(valid_values))

        # Compute absolute threshold
        threshold_absolute = baseline_median * (threshold_percent / 100.0)

        # Identify fluctuations
        fluctuation_indices = []
        fluctuation_details = []

        for value, proj_idx in valid_pairs:
            deviation = value - baseline_median
            deviation_percent = (deviation / baseline_median) * 100.0

            if abs(deviation) > threshold_absolute:
                fluctuation_indices.append(proj_idx)
                fluctuation_details.append(
                    {
                        "projection_i": proj_idx,
                        "value": round(value, 4),
                        "deviation_from_baseline": round(deviation, 4),
                        "deviation_percent": round(deviation_percent, 2),
                    }
                )

        num_stable = len(valid_pairs) - len(fluctuation_details)

        return {
            "baseline_median": round(baseline_median, 4),
            "num_measurements": len(values),
            "num_valid": len(valid_pairs),
            "num_stable": num_stable,
            "num_fluctuations": len(fluctuation_details),
            "fluctuation_threshold_percent": threshold_percent,
            "fluctuation_indices": fluctuation_indices,
            "fluctuation_details": fluctuation_details,
        }

    async def _process_flat_fields(self, name: str, producer: AsyncIterator[ArrayLike]) -> None:
        """
        Accumulates dark and flat fields, average them and store as dynamic class members, _dark,
        _flat
        """
        num_proj = 0
        buffer: List[ArrayLike] = []
        async for projection in producer:
            buffer.append(projection)
            num_proj += 1
        setattr(self, name, np.array(buffer).mean(axis=0))
        self.info_stream(
            "%s: processed %d %s projections",
            self.__class__.__name__,
            num_proj,
            name[1:],
        )

    @DebugIt()
    @command()
    async def update_darks(self) -> None:
        await self._process_stream(self._process_flat_fields("_dark", self._receiver.subscribe()))

    @DebugIt()
    @command()
    async def update_flats(self) -> None:
        await self._process_stream(self._process_flat_fields("_flat", self._receiver.subscribe()))

    @DebugIt()
    @command(dtype_in=str)
    async def estimate_spatial_resolution(self, path: str) -> None:
        """Computes FRC for resolution estimation from radiogram projections"""
        await self._process_stream(
            self._estimate_spatial_resolution(self._receiver.subscribe(), path)
        )

    async def _estimate_spatial_resolution(
        self, producer: AsyncIterator[ArrayLike], path: str
    ) -> None:
        """
        Compute Fourier Ring Correlation between consecutive projections with configurable offset.
        Results are stored as JSON file in specified path.

        :param producer: asynchronous generator of projections.
        :type producer: AsyncIterator[ArrayLike]
        :param path: directory where results will be saved
        :type path: str
        """
        correlation_results: List[Dict[str, Any]] = []
        classical_resolutions: List[Optional[float]] = []
        geometric_resolutions: List[Optional[float]] = []
        proj_indices: List[int] = []
        proj_idx: int = 0
        previous_proj: Optional[ArrayLike] = None
        crop_determined = False
        with PerformanceTracker():
            try:
                async for proj in producer:
                    # Determine crop region from first projection.
                    if not crop_determined:
                        bbox_height = self._y_end - self._y_start
                        use_fallback = (
                            (self._y_start == 0 and self._y_end == 0) or 
                            bbox_height < self._crop_height
                        )
                        if use_fallback:
                            y_start, y_end, _ = select_frc_region(
                                flat_correct(proj, self._flat, self._dark),
                                crop_height=self._crop_height,
                                padding_y=self._padding_y,
                                padding_x=self._padding_x,
                            )
                            self._y_start = y_start
                            self._y_end = y_end
                            self.info_stream(
                                "%s: FRC crop region: auto-determined y=[%d:%d], (height=%d)",
                                self.__class__.__name__,
                                self._y_start, self._y_end, self._crop_height)
                        else:
                            crop_start, crop_end = self._compute_centered_crop(
                                self._y_start, self._y_end
                            )
                            self._y_start = crop_start
                            self._y_end = crop_end
                            self.info_stream(
                                "%s: FRC crop region: specified y=[%d:%d], (height=%d)",
                                self.__class__.__name__,
                                self._y_start, self._y_end, self._crop_height)
                        crop_determined = True
                    self.info_stream(
                        "%s: FRC crop region: determined y=[%d:%d], (height=%d)",
                        self.__class__.__name__,
                        self._y_start, self._y_end, self._crop_height)
                    # Initialize state for FRC by computing the frequency beans.
                    if not self._frc_state:
                        self._frc_state = prepare_frc_state(self._crop_height, proj.shape[1])
                        self.info_stream(
                            "%s: FRC state initialized for dim: [%d x %d]",
                            self.__class__.__name__,
                            self._crop_height,
                            proj.shape[1],
                        )
                    # When proj_idx = 0 i.e. we are processing the very first projection, previous
                    # projection is also None. In that case we will only store current projection
                    # for correlating later on.
                    if previous_proj is not None:
                        proj_pair_start_idx = proj_idx - 1
                        # Start correlating if offset is satisfied
                        if proj_pair_start_idx % self._proj_offset == 0:
                            self.info_stream(
                                "%s: correlating: proj: %d to proj: %d",
                                self.__class__.__name__,
                                proj_pair_start_idx,
                                proj_pair_start_idx + 1,
                            )
                            result = compute_frc(
                                previous_proj[self._y_start : self._y_end, :],
                                proj[self._y_start : self._y_end, :],
                                self._dark[self._y_start : self._y_end, :],
                                self._flat[self._y_start : self._y_end, :],
                                frc_state=self._frc_state,
                                threshold_method=self._resolution_threshold,
                            )
                            json_result = self._result_to_json_dict(
                                result, proj_idx - self._proj_offset, proj_idx
                            )
                            correlation_results.append(json_result)
                            # Collect for statistics
                            classical_resolutions.append(json_result["classical_resolution"])
                            geometric_resolutions.append(json_result["geometric_resolution"])
                            proj_indices.append(proj_idx - self._proj_offset)
                    previous_proj = proj
                    proj_idx += 1

                # Compute outlier-aware statistics
                classical_stats = self._compute_outlier_aware_statistics(
                    classical_resolutions,
                    proj_indices,
                    threshold_percent=self._fluctuation_threshold,
                )

                geometric_stats = self._compute_outlier_aware_statistics(
                    geometric_resolutions,
                    proj_indices,
                    threshold_percent=self._fluctuation_threshold,
                )

                summary_statistics = {
                    "classical_resolution": classical_stats,
                    "geometric_resolution": geometric_stats,
                }

                # Build final JSON structure.
                # At this point the absolute path looks like ../acq_id/radios because we passed in the
                # current directory, which walker points to. Hence after splitting we need to grab the
                # second last component.
                acquisition_id = os.path.normpath(path).split("/")[-2]
                acquisition_path = "/".join(os.path.normpath(path).split("/")[:-1])
                payload = {
                    "acquisition_id": acquisition_id,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z",
                    "configuration": {
                        "offset": self._proj_offset,
                        "threshold_method": self._resolution_threshold,
                        "num_projections_received": proj_idx,
                        "num_frc_computations": len(correlation_results),
                        "crop_y_start": self._y_start,
                        "crop_y_end": self._y_end,
                    },
                    "summary_statistics": summary_statistics,
                    "correlation_results": correlation_results,
                }

                # Construct full file path
                json_filepath = os.path.join(acquisition_path, f"{acquisition_id}_frc.json")

                # Write JSON file
                try:
                    assert self._walker
                    await self._walker.log_to_json(
                        payload=json.dumps(payload, indent=4), filename=json_filepath
                    )
                    self.info_stream(
                        "%s: Saved FRC results to %s (%d measurements)",
                        self.__class__.__name__,
                        json_filepath,
                        len(correlation_results),
                    )
                except Exception as e:
                    self.error_stream(
                        "%s: Failed to write FRC JSON: %s. Continuing anyway.",
                        self.__class__.__name__, str(e))
            except Exception as e:
                self.error_stream("%s: FRC computation failed: %s",
                                  self.__class__.__name__, str(e))
                raise