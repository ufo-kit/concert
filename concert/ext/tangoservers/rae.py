"""
qa.py
-----
Implements a device server to execute quality assurance routines during acquisition.
"""
import copy
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
from concert.measures import rotation_axis


class EstimationAlgorithm(IntEnum):
    """Enumerates among the choices for the estimation algorithm"""
    MARKER_TRACKING = 0
    PHASE_CORRELATION = 1


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

    axis_of_rotation = attribute(
        label="Axis of rotation",
        dtype=(float,),
        max_dim_x=3,
        access=AttrWriteType.READ_WRITE,
        fget="get_axis_of_rotation",
        fset="set_axis_of_rotation",
        doc="encapsulates center of rotation and the angular corrections along y-axis(roll_angle) \
        and x-axis(tilt_angle), [center_of_rotation, roll_angle, tilt_agle]"
    )

    meta_attr_mt = attribute(
        label="Meta attributes for marker tracking",
        dtype=(int,),
        max_dim_x=6,
        access=AttrWriteType.WRITE,
        fset="set_meta_attr_mt",
        doc="encapsulates meta attributes for marker tracking i.e. crop_vertical, crop_left, \
        crop_right, num_markers, marker_radius, patch_width"
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

    async def init_device(self) -> None:
        await super().init_device()
        self._norm_buffer = []
        self._reference_sinogram = np.array([])
        self._estimation_algorithm = EstimationAlgorithm.MARKER_TRACKING
        self._axis_of_rotation = np.zeros((3,))
        self.set_state(DevState.STANDBY)
        self.info_stream("%s: init_device: %s", self.__class__.__name__, self.get_state())

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

    def get_axis_of_rotation(self) -> ArrayLike:
        return self._axis_of_rotation

    def set_axis_of_rotation(self, new_value: ArrayLike) -> None:
        self._axis_of_rotation = new_value

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
        num_frame = 0
        buffer: List[ArrayLike] = []
        crop_vertical, left, crop_right = self._meta_attr_mt[:3]
        async for projection in producer:
            right: int = projection.shape[1] - crop_right
            if crop_vertical > 0:
                top: int = 0
                bottom: int = projection.shape[0] // crop_vertical 
            else:
                top: int = projection.shape[0] // crop_vertical
                bottom: int = -1
            patch: ArrayLike = projection[top:bottom, left:right]
            buffer.append(patch)
            num_frame += 1
        setattr(self, name, np.array(buffer).mean(axis=0))
        self.info_stream("%s:processed %d %s frames", self.__class__.__name__, num_frame, name[1:])

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
    async def estimate_axis_of_rotation(self) -> None:
        """Estimates the center of rotation"""
        if self._estimation_algorithm == EstimationAlgorithm.MARKER_TRACKING:
            await self._process_stream(self._estimate_marker_tracking(self._receiver.subscribe()))
        else:
            await self._process_stream(self._estimate_phase_corr(self._receiver.subscribe()))

    @staticmethod
    def opt_func(angle_x: np.float64, center_p1: np.float64, radius_p2: np.float64,
                 phase_p3: np.float64) -> np.float64:
        """Defines the model function for nonlinear polynomial fit"""
        return center_p1 + radius_p2 * np.cos(angle_x + phase_p3)

    def _track_merkers(self, patch: ArrayLike, up_samp_corr: bool = False, **kwargs) -> ArrayLike:
        """
        We use normalized cross correlation to determine the highest signal response with respect to
        pre-defined absorption pattern of a sphere.
        We simulate this absorption pattern with the assumption that it is not uniform everywhere.
        Highest absorption takes place in the middle, where we encounter the whole diametric width
        along the path of the beam and it reduces as we move away from the center.
        
        Meshgrid gives us with incremental range of values, which when squared tend to be maximum at
        the edge gradually reducing to minimum as we move to the middle. This can then be used to
        simulate approximate absorption pattern by a sphere. We can then perform a normalized
        cross-correlation and track the locations having highest signal response.
        """
        patch: ArrayLike = copy.deepcopy(patch)
        centroids: List[ArrayLike] = []
        if up_samp_corr:
            up_sample_by: float = kwargs.get("up_sampled_by", 1.25)
            patch = smt.rescale(image=patch, scale=up_sample_by)
        try:
            radius, width = self._meta_attr_mt[-2:]
            y, x = np.mgrid[-radius:radius+1, -radius:radius+1]
            mask = np.where(radius ** 2 - x ** 2 - y ** 2 >= 0)
            sphere = np.zeros((2 * radius + 1, 2 * radius + 1))
            sphere[mask] = 2 * np.sqrt(radius ** 2 - x[mask] ** 2 - y[mask] ** 2)
            corr: ArrayLike = nft.ifft2(nft.fft2(patch - patch.mean()) * np.conjugate(
                nft.fft2(sphere - sphere.mean(), s=patch.shape))).real
            ym_1, xm_1 = np.unravel_index(np.argmax(corr), corr.shape)
            corr[ym_1-width:ym_1+width, xm_1-width:xm_1+width] = 0
            ym_2, xm_2 = np.unravel_index(np.argmax(corr), corr.shape)
            ym_1 += radius
            xm_1 += radius
            ym_2 += radius
            xm_2 += radius
            centroids: List[List[int]] = [[ym_1, xm_1], [ym_2, xm_2]]
            centroids: ArrayLike = np.array(sorted(centroids, key=lambda centroid: centroid[0]))
            mp1: ArrayLike = patch[centroids[0,0]-width:centroids[0,0]+width,
                                   centroids[0,1]-width:centroids[0,1]+width]
            mp2: ArrayLike = patch[centroids[1,0]-width:centroids[1,0]+width,
                                   centroids[1,1]-width:centroids[1,1]+width]
        except Exception as e:
            self.info_stream("%s:runtime tracking error: %s", self.__class__.__name__, e.__str__())
            raise e
        return centroids, mp1, mp2

    async def _estimate_marker_tracking(self, producer: AsyncIterator[ArrayLike]) -> None:
        """
        Estimates the center of rotation with marker tracking and nonlinear polynomial fit.
        """
        # Extract all tracking and estimation parameters
        crop_vertical, left, crop_right, num_markers, _, _ = self._meta_attr_mt
        wait_window, check_window, offset = self._meta_attr_mt_estm
        beta, grad_thresh = self._meta_attr_mt_eval
        det_row_idx, _ = self._meta_attr_phase_corr
        # Initialize local variables
        projection_count = 0
        ref_sino_buffer: List[ArrayLike] = []
        marker_centroids: List[ArrayLike] = []
        estm_axis: List[float] = []
        estm_axis_angle_y: List[float] = []
        estm_axis_angle_x: List[float] = []
        event_triggered = False
        optimal_marker = -1
        ffc = FlatCorrect(dark=self._dark, flat=self._flat, absorptivity=True)
        # TODO: Bring back ffc(producer)
        async for projection in producer:
            right: int = projection.shape[1] - crop_right
            if crop_vertical > 0:
                top: int = 0
                bottom: int = projection.shape[0] // crop_vertical 
            else:
                top: int = projection.shape[0] // crop_vertical
                bottom: int = -1
            projection_count += 1
            # Prepare for phase correlation for the next scan for all projections.
            ref_sino_buffer.append(projection[det_row_idx, :])
            if not event_triggered:
                patch: ArrayLike = projection[top:bottom, left:right]
                # TODO: Remove manual flat field correction
                trans = np.nan_to_num((patch - self._dark) / (self._flat - self._dark), nan=0.0,
                                      posinf=0.0, neginf=0.0)
                patch = -np.log2(trans, where=trans>0)
                centroids, _, _ = self._track_merkers(patch)
                marker_centroids.append(centroids)
                # Initial wait window is used to select an optimal marker and prevent runtime error
                # in polynomial fit
                if projection_count < wait_window:
                    self.info_stream(
                        "%s:skipping estimation [%d/%d]",self.__class__.__name__, projection_count,
                        wait_window)
                else:
                    if optimal_marker < 0:
                        vert_disp_err: List[float] = []
                        for mix in range(num_markers):
                            vert_disp_err.append(np.std(np.array(marker_centroids)[:, mix, 0]))
                        optimal_marker = np.argmin(vert_disp_err)
                        #TODO: Remove optimal marker later
                        optimal_marker = 0
                        self.info_stream("%s:optimal marker idx: %d", self.__class__.__name__,
                                         optimal_marker)
                    if projection_count % offset == 0:
                        try:
                            x: ArrayLike = self._angle_dist[:projection_count]
                            y: ArrayLike = np.array(marker_centroids)[:, optimal_marker, 1]
                            params, _ = sop.curve_fit(f=self.opt_func, xdata=np.squeeze(x),
                                                      ydata=np.squeeze(y))
                            # Params[0] from curve-fit is the estimated axis w.r.t cropped patch.
                            roll_y, tilt_x, _ = rotation_axis(
                                    np.array(marker_centroids)[:, optimal_marker])
                            self.info_stream(
                                    "%s:[%d/%d] estimated: [center: %f, roll_y: %f, tilt_x: %f]",
                                    self.__class__.__name__, projection_count, self._num_radios,
                                    left + params[0], roll_y, tilt_x)
                            # Use runtime averaging on the estimated values.
                            if len(estm_axis) == 0:
                                estm_axis.append(left + params[0])
                                estm_axis_angle_y.append(roll_y)
                                estm_axis_angle_x.append(tilt_x)
                            else:
                                avg_estm: float = (beta * estm_axis[-1]) + ((1 - beta) * (
                                    left + params[0]))
                                avg_axis_angle_y: float = (beta * estm_axis_angle_y[-1]) + (
                                        (1 - beta) * roll_y)
                                avg_axis_angle_x: float = (beta * estm_axis_angle_x[-1]) + (
                                        (1 - beta) * tilt_x)
                                avg_estm /= (1 - beta**projection_count)
                                avg_axis_angle_y /= (1 - beta**projection_count)
                                avg_axis_angle_x /= (1 - beta**projection_count)
                                estm_axis.append(avg_estm)
                                estm_axis_angle_y.append(avg_axis_angle_y)
                                estm_axis_angle_x.append(avg_axis_angle_x)
                            if len(estm_axis) > check_window:
                                # TODO: Monitor window is used for troubleshooting. Remove later.
                                if projection_count % self._monitor_window == 0:
                                    self.info_stream("absolute gradient:\n%s", np.abs(np.gradient(estm_axis)))
                                abs_grad: ArrayLike = np.abs(np.gradient(estm_axis[-check_window:]))
                                if np.all(abs_grad < grad_thresh):
                                    # Set the last estimated value as the final center of rotation
                                    self._axis_of_rotation = np.array([estm_axis[-1],
                                                                       estm_axis_angle_y[-1],
                                                                       estm_axis_angle_x[-1]])
                                    self.info_stream("%s:final: %s", self.__class__.__name__,
                                                     self._axis_of_rotation)
                                    self.push_event("axis_of_rotation", [], [],
                                                    self._axis_of_rotation)
                                    event_triggered = True
                        except RuntimeError:
                            self.info_stream(
                                "%s could not find optimal parameters with projection: [%d/%d]",
                                self.__class__.__name__, projection_count, self._num_radios)
        # Store the reference sinogram to be used for phase correlation during subsequent scan
        self._reference_sinogram = np.vstack(ref_sino_buffer)
        self.info_stream("%s:generated reference sinogram", self.__class__.__name__)

    async def _estimate_phase_corr(self, producer: AsyncIterator[ArrayLike]) -> None:
        """Estimates center of rotation with correlation with phase correlation"""
        # We assert, that there is a pre-computed known value exists for the center of rotation
        # because this method estimates the potential error for the same.
        assert(self._axis_of_rotation is not None and self._axis_of_rotation != 0.)
        # We assert that the reference sinogram from previous measurement is available for the
        # cross correlation.
        assert(self._reference_sinogram is not None)
        det_row_idx, num_proj_corr = self._meta_attr_phase_corr
        moving_sino: List[ArrayLike] = []
        ffc = FlatCorrect(dark=self._dark, flat=self._flat, absorptivity=True)
        projection_count = 0
        self.info_stream("%s:commencing axis correction estimation with phase correlation",
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
        self._axis_of_rotation += corr_peak_loc[1]
        self.info_stream("%s: estimated axis error %d pixles, revised center of rotation: %d",
                         self.__class__.__name__, axis_correction, self._axis_of_rotation)
        self.push_event("axis_of_rotation", [], [], self._axis_of_rotation)


if __name__ == "__main__":
    pass
