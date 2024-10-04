"""
qa.py
-----
Implements a device server to execute quality assurance routines during acquisition.
"""
from enum import IntEnum
from typing import List, AsyncIterator, Tuple, Dict, Awaitable, Optional
import numpy as np
import numpy.fft as nft
from numpy.typing import ArrayLike
import pandas as pd
import scipy.ndimage as snd
import scipy.optimize as sop
import scipy.signal as scs
import skimage.filters as skf
import skimage.measure as sms
import skimage.feature as smf
import skimage.transform as smt
from skimage.measure._regionprops import RegionProperties
from tango import DebugIt, DevState, CmdArgType, EventType, ArgType, AttrDataFormat
from tango.server import attribute, command, AttrWriteType, pipe, PipeWriteType
from concert.ext.tangoservers.base import TangoRemoteProcessing
from concert.ext.ufo import FlatCorrect, GaussianFilter, MedianFilter


class EstimationAlgorithm(IntEnum):
    """Enumerates among the choices for the estimation strategies"""

    MT_SEGMENTATION = 0
    PHASE_CORRELATION = 1
    REF_SINOGRAM = 2


class RotationAxisEstimator(TangoRemoteProcessing):
    """
    Implements Tango device server to estimate axis of rotation.
    """
    rot_angle = attribute(
        label="Overall angle of rotation",
        dtype=float,
        access=AttrWriteType.READ_WRITE,
        fget="get_rot_angle",
        fset="set_rot_angle",
        doc="overall angle of rotation for current acquisition"
    )

    num_darks = attribute(
        label="Number of darks",
        dtype=int,
        access=AttrWriteType.READ_WRITE,
        fget="get_num_darks",
        fset="set_num_darks",
        doc="overall number of dark field acquisitions"
    )

    num_flats = attribute(
        label="Number of flats",
        dtype=int,
        access=AttrWriteType.READ_WRITE,
        fget="get_num_flats",
        fset="set_num_flats",
        doc="overall number of flat field acquisitions"
    )

    num_radios = attribute(
        label="Number of projections",
        dtype=int,
        access=AttrWriteType.READ_WRITE,
        fget="get_num_radios",
        fset="set_num_radios",
        doc="overall number of projections to be acquired"
    )

    center_of_rotation = attribute(
        label="Center of rotation",
        dtype=float,
        access=AttrWriteType.READ_WRITE,
        fget="get_center_of_rotation",
        fset="set_center_of_rotation",
        doc="estimated or known center of rotation, when phase correlation or image registration \
        methods are used to estimate a shift in center, we need previously known value to correct"
    )

    meta_attr_mt = attribute(
        label="Meta attributes for marker tracking",
        dtype=(int,),
        max_dim_x=6,
        access=AttrWriteType.WRITE,
        fset="set_meta_attr_mt",
        doc="encapsulates meta attributes for marker tracking i.e. crop_top, crop_bottom, \
        crop_left, crop_right, num_markers, avg_window"
    )

    meta_attr_mt_estm = attribute(
        label="Meta attributes for axis estimation",
        dtype=(int,),
        max_dim_x=3,
        access=AttrWriteType.WRITE,
        fset="set_meta_attr_mt_estm",
        doc="encapsulates attributes for axis estimation using marker tracking i.e., initial \
        number of projections to wait before starting estimation (wait_window), number of \
        projections to check the stability in estimated value(check_window), a projection offset \
        to run curve fitting (offset)"
    )

    meta_attr_mt_eval = attribute(
        label="Meta attributes for evaluation",
        dtype=(float,),
        max_dim_x=2,
        access=AttrWriteType.WRITE,
        fset="set_meta_attr_mt_eval",
        doc="beta parameter for smoothing estimated axis values(beta), threshold for the absolute \
        gradient to detect the plateau in estimated axis values(grad_thresh)"
    )

    meta_attr_phase_corr = attribute(
        label="Meta attributes for phase correlation",
        dtype=(int,),
        max_dim_x=2,
        access=AttrWriteType.WRITE,
        fset="set_meta_attr_phase_corr",
        doc="encapsulates meta attributes for phase correlation i.e., detector row index for \
        feature to correlate(det_row_idx), number of projection for correlation(num_proj_corr)"
    )

    _angle_dist: ArrayLike
    _norm_buffer: List[ArrayLike]
    _estimation_algorithm: EstimationAlgorithm
    _reference_sinogram: ArrayLike
    _monitor_window: int = 200 # TODO: Temporary internal attribute, remove when ready
    _last_detected_marker_centroids: Optional[ArrayLike]

    async def init_device(self) -> None:
        await super().init_device()
        self._norm_buffer = []
        self._reference_sinogram = np.array([])
        self._estimation_algorithm = EstimationAlgorithm.MT_SEGMENTATION
        self._center_of_rotation = None
        self._last_detected_marker_centroids = None
        self.set_state(DevState.STANDBY)
        self.info_stream("%s initialized device with state: %s",
                         self.__class__.__name__, self.get_state())

    @attribute
    def estimation_algorithm(self) -> EstimationAlgorithm:
        return self._estimation_algorithm

    @estimation_algorithm.setter
    def estimation_algorithm(self, est_algo: int) -> None:
        self._estimation_algorithm = EstimationAlgorithm(est_algo)
        self.info_stream("%s: using estimation algorithm %s", self.__class__.__name__,
                         self._estimation_algorithm.name)

    def get_rot_angle(self) -> float:
        return self._rot_angle

    def set_rot_angle(self, rot_angle: float) -> None:
        self._rot_angle = rot_angle
        self.info_stream(
            "%s has set overall angle of rotation to: %f, state: %s",
            self.__class__.__name__, self._rot_angle, self.get_state()
        )

    def get_num_darks(self) -> int:
        return self._num_darks

    def set_num_darks(self, num_darks: int) -> int:
        self._num_darks = num_darks
        self.info_stream(
            "%s has set overall number of dark field acquisitions to: %d, state: %s",
            self.__class__.__name__, self._num_darks, self.get_state()
        )

    def get_num_flats(self) -> int:
        return self._num_flats

    def set_num_flats(self, num_flats: int) -> None:
        self._num_flats = num_flats
        self.info_stream(
            "%s has set overall number of flat field acquisitions to: %d, state: %s",
            self.__class__.__name__, self._num_flats, self.get_state()
        )

    def get_num_radios(self) -> int:
        return self._num_radios

    def set_num_radios(self, num_radios: int) -> None:
        self._num_radios = num_radios
        self.info_stream(
            "%s has set overall number of projections to: %d, state: %s",
            self.__class__.__name__, self._num_radios, self.get_state()
        )

    def get_center_of_rotation(self) -> float:
        return self._center_of_rotation

    def set_center_of_rotation(self, new_value: float) -> None:
        self._center_of_rotation = new_value

    def set_meta_attr_mt(self, mam: ArrayLike) -> None:
        self._meta_attr_mt = mam

    def set_meta_attr_mt_estm(self, mame: ArrayLike) -> None:
        self._meta_attr_mt_estm = mame

    def set_meta_attr_mt_eval(self, mamv: ArrayLike) -> None:
        self._meta_attr_mt_eval = mamv

    def set_meta_attr_phase_corr(self, mapc: ArrayLike) -> None:
        self._meta_attr_phase_corr = mapc

    @DebugIt()
    @command()
    async def prepare_angular_distribution(self) -> None:
        """Prepares projected angular distribution as input values for nonlinear polynomial fit"""
        self._angle_dist = np.linspace(0., self._rot_angle, self._num_radios)
        self.info_stream("%s: prepared angular distribution", self.__class__.__name__)

    async def _update_ff_projections(self, name: str, producer: AsyncIterator[ArrayLike]) -> None:
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
                "%s: processed %d %s projections", self.__class__.__name__, num_proj, name[1:])

    @DebugIt()
    @command()
    async def update_darks(self) -> None:
        await self._process_stream(self._update_ff_projections("_dark", self._receiver.subscribe()))

    @DebugIt()
    @command()
    async def update_flats(self) -> None:
        await self._process_stream(self._update_ff_projections("_flat", self._receiver.subscribe()))

    @DebugIt()
    @command()
    async def estimate_center_of_rotation(self) -> None:
        """Estimates the center of rotation"""
        if self._estimation_algorithm == EstimationAlgorithm.MT_SEGMENTATION:
            await self._process_stream(self._estimate_marker_tracking(self._receiver.subscribe()))
        elif self._estimation_algorithm == EstimationAlgorithm.PHASE_CORRELATION:
            await self._process_stream(self._estimate_phase_corr(self._receiver.subscribe()))
        else:
            await self._process_stream(self._derive_reference_sinogram(self._receiver.subscribe()))

    @staticmethod
    def opt_func(angle_x: np.float64, center_p1: np.float64, radius_p2: np.float64,
                 phase_p3: np.float64) -> np.float64:
        """Defines the model function for nonlinear polynomial fit"""
        return center_p1 + radius_p2 * np.cos(angle_x + phase_p3)

    def _extract_marker_centroids(self, patch: ArrayLike, num_markers: int, avg_window: int,
                                  bins: int = 128) -> ArrayLike:
        """
        Uses histogram segmentation to extract marker centroids.
        Log transform and subsequent smoothing are used to make the histogram of the patch
        approximately normal. Patch is selected to localize the markers along with a limited part
        of the projection where absorption is relatively lower compared to the markers. This results
        into the histogram having at least two local maximas and minimas.

        Local minima and maxima towards the right edge associated with higher signal intensities
        correspond to the markers, since we are working with absorption projections. We are
        interested with bin edge intensity associated with this local minima, because it can be
        used as a threshold to create a mask that separates the markers from everything else based
        their higher absorption. For each iteration we store the last detected marker centroids to
        deal with the occassional anomalies during segmentation. If segmentation does not result
        into optimal marker centroids, we simply reuse the last observed centroids. This can be done
        because marker displacement for two subsequent projections is very small, usually less than
        one pixel.
        """
        centroids: List[ArrayLike] = []
        try:
            hist, bin_edges = np.histogram(patch, bins=bins)
            # We constrain the log transform to non-zero values only to avoid runtime issues.
            log_hist: ArrayLike = np.log2(hist, where=hist > 0)
            avg_log_hist: ArrayLike = pd.Series(log_hist).rolling(window=avg_window,
                                                                  center=True).mean().values
            loc_min_idx: ArrayLike = scs.argrelextrema(avg_log_hist, np.less)[0]
            mask: ArrayLike = patch < bin_edges[loc_min_idx[-1]]
            labels: ArrayLike = sms.label(~mask)
            regions: List[RegionProperties] = sms.regionprops(label_image=labels)
            centroids = [region.centroid for region in regions]
            # Markers centroids are sorted according to vertical location for consistency across
            # projections.
            centroids: ArrayLike = np.array(sorted(centroids, key=lambda centroid: centroid[0]))
            if len(centroids) != num_markers:
                if self._last_detected_marker_centroids is None:
                    raise RuntimeError("marker segmentation anomaly on first projection")
                centroids = self._last_detected_marker_centroids
            else:
                self._last_detected_marker_centroids = centroids
            assert(len(centroids) == num_markers)
        except IndexError:
            centroids = self._last_detected_marker_centroids
        return centroids

    async def _estimate_marker_tracking(self, producer: AsyncIterator[ArrayLike]) -> None:
        """
        Estimates the center of rotation with marker tracking and nonlinear polynomial fit.
        """
        projection_count = 0
        marker_centroids: List[ArrayLike] = []
        estm_axis: List[float] = []
        event_triggered = False
        crop_top, crop_bottom, crop_left, crop_right, num_markers, avg_window = self._meta_attr_mt
        wait_window, check_window, offset = self._meta_attr_mt_estm
        beta, grad_thresh = self._meta_attr_mt_eval
        optimal_marker = -1
        det_row_idx, _ = self._meta_attr_phase_corr
        ffc = FlatCorrect(dark=self._dark, flat=self._flat, absorptivity=True)
        ref_sino_buffer: List[ArrayLike] = []
        async for projection in ffc(producer):
            projection_count += 1
            # Prepare for phase correlation for the next scan for all projections.
            ref_sino_buffer.append(projection[det_row_idx, :])
            if not event_triggered:
                patch: ArrayLike = projection[crop_top:crop_bottom, crop_left: crop_right]
                marker_centroids.append(self._extract_marker_centroids(patch, num_markers,
                                                                       avg_window))
                # Initial wait window is introduced for the optimization method to be able to yield
                # some values for the parameters. We also use this wait window to evaluate an
                # optimal marker to limit the scope of curve fit.
                if projection_count < wait_window:
                    self.info_stream(
                        "%s: Skipping estimation until [%d/%d]",
                        self.__class__.__name__,
                        projection_count,
                        wait_window
                    )
                else:
                    if optimal_marker < 0:
                        vrt_dsp_stderr: List[float] = []
                        for mix in range(num_markers):
                            vrt_dsp_stderr.append(np.std(np.array(marker_centroids)[:, mix, 0]))
                            optimal_marker = np.argmin(vrt_dsp_stderr)
                    if projection_count % offset == 0:
                        try:
                            x: ArrayLike = self._angle_dist[:projection_count]
                            y: ArrayLike = np.array(marker_centroids)[:, optimal_marker, 1]
                            params, _ = sop.curve_fit(f=self.opt_func,
                                                      xdata=np.squeeze(x),
                                                      ydata=np.squeeze(y))
                            # Params[0] from curve-fit is the estimated axis of rotation w.r.t
                            # each marker. It depends upon the order of optimizable parameters,
                            # which is used to define the optimization function.
                            self.info_stream(
                                    "%s: Estimated rotation axis with projections [%d/%d]: %f",
                                    self.__class__.__name__,
                                    projection_count,
                                    self._num_radios,
                                    crop_left + params[0]
                                )
                            # We observe the change in this value w.r.t a configurable error
                            # threshold over a time frame in terms of number of projections, given
                            # by check window.
                            # The _monitor_window property is used only for logging purpose.
                            # Ideally, we would remove it upon reaching a stable implementation.
                            if len(estm_axis) == 0:
                                estm_axis.append(crop_left + params[0])
                            else:
                                avg_estm: float = (beta * estm_axis[-1]) + (
                                        (1 - beta) * (crop_left + params[0]))
                                avg_estm /= (1 - beta**projection_count)
                                estm_axis.append(avg_estm)
                            if len(estm_axis) > check_window:
                                # TODO: Remove following if condition entirely before merge along with
                                # self._monitor_window
                                if projection_count % self._monitor_window == 0:
                                    self.info_stream("absolute gradient:\n%s", np.abs(
                                        np.gradient(estm_axis)))
                                abs_grad: ArrayLike = np.abs(np.gradient(estm_axis[-check_window:]))
                                if np.all(abs_grad < grad_thresh):
                                    # Set the last estimated value as the final center of rotation
                                    self._center_of_rotation = estm_axis[-1]
                                    self.info_stream("%s:estimated final center of rotation %f.",
                                                     self.__class__.__name__,
                                                     self._center_of_rotation)
                                    self.push_event("center_of_rotation", [], [],
                                                           self._center_of_rotation)
                                    event_triggered = True
                                    # Reset the marker centroids for subsequent estimation.
                                    self._last_detected_marker_centroids = None
                        except RuntimeError:
                            self.info_stream(
                                "%s could not find optimal parameters with projection: [%d/%d]",
                                self.__class__.__name__, projection_count, self._num_radios)
            # Store the reference sinogram to be used for phase correlation during subsequent scan
        self._reference_sinogram = np.vstack(ref_sino_buffer)
        self.info_stream("%s: stored reference sinogram for phase correlation.",
                         self.__class__.__name__)

    async def _estimate_phase_corr(self, producer: AsyncIterator[ArrayLike]) -> None:
        """Estimates center of rotation with correlation with phase correlation"""
        # We assert, that there is a pre-computed known value exists for the center of rotation
        # because this method estimates the potential error for the same.
        assert(self._center_of_rotation is not None and self._center_of_rotation != 0.)
        # We assert that the reference sinogram from previous measurement is available for the
        # cross correlation.
        assert(self._reference_sinogram is not None)
        det_row_idx, num_proj_corr = self._meta_attr_phase_corr
        moving_sino: List[ArrayLike] = []
        ffc = FlatCorrect(dark=self._dark, flat=self._flat, absorptivity=True)
        projection_count = 0
        self.info_stream("%s: commencing axis correction estimation with phase correlation",
                         self.__class__.__name__)
        async for projection in ffc(producer):
            projection_count += 1
            if projection_count < num_proj_corr:
                moving_sino.append(projection[det_row_idx, :])
            else:
                break
        moving_sino_arr: ArrayLike = np.vstack(moving_sino)
        corr: ArrayLike = nft.ifft2(
                nft.fft2(
                    self._reference_sinogram - self._reference_sinogram.mean()) * np.conj(nft.fft2(
                        moving_sino_arr - moving_sino_arr.mean(), s=self._reference_sinogram.shape))
                    ).real
        corr_peak_loc: Tuple[int, int] = np.unravel_index(corr.argmax(), corr.shape)
        # Phase correlation using normalized cross correlation yields a signal peak value in spatial
        # domain. Horizontal coordinate of this peak is the error from previous estimate, which we
        # need to correct.
        self._center_of_rotation += corr_peak_loc[1]
        self.info_stream("%s: estimated axis error %d pixles, revised center of rotation: %d",
                         self.__class__.__name__, axis_correction, self._center_of_rotation)
        self.push_event("center_of_rotation", [], [], self._center_of_rotation)

    async def _derive_reference_sinogram(self, producer: AsyncIterator[ArrayLike]) -> None:
        """Estimates center of rotation with image registration and gradient descent optimization"""
        det_row_idx, _ = self._meta_attr_phase_corr
        ffc = FlatCorrect(dark=self._dark, flat=self._flat, absorptivity=True)
        ref_sino_buffer: List[ArrayLike] = []
        self.info_stream("%s: deriving reference sinogram", self.__class__.__name__)
        async for projection in ffc(producer):
            ref_sino_buffer.append(projection[det_row_idx, :])
        self._reference_sinogram = np.vstack(ref_sino_buffer)
        self.info_stream("%s:triggering event with precomputed center of rotation%f.",
                         self.__class__.__name__, self._center_of_rotation)
        self.push_event("center_of_rotation", [], [], self._center_of_rotation)


if __name__ == "__main__":
    pass
