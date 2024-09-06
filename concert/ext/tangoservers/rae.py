"""
qa.py
-----
Implements a device server to execute quality assurance routines during acquisition.
"""
from enum import IntEnum
from typing import List, AsyncIterator, Tuple, Dict, Awaitable
import numpy as np
from numpy.typing import ArrayLike
import scipy.ndimage as snd
import scipy.optimize as sop
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
    MT_HOUGH_TRANSFORM = 1
    OPT_GRADIENT_DESCENT = 2


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

    estm_offset = attribute(
        label="Estmation offset",
        dtype=int,
        access=AttrWriteType.READ_WRITE,
        fget="get_estm_offset",
        fset="set_estm_offset",
        doc="offset in number of projections to be used to run estimation"
    )

    center_of_rotation = attribute(
        label="Center of rotation",
        dtype=float,
        access=AttrWriteType.READ_WRITE,
        fget="get_center_of_rotation",
        fset="set_center_of_rotation",
        doc="estimated center of rotation"
    )

    marae = attribute(
        label="Meta attributes for rotation axis estimation",
        dtype=(int,),
        max_dim_x=5,  # max_dim_x corresponds to the number of elements packed into it
        access=AttrWriteType.WRITE,
        fset="set_marae",
        doc="encapsulates meta attributes for rotation axis estimation, crop_left, crop_right, \
        crop_vertical, num_markers, marker_diameter_px"
    )

    _angle_dist: ArrayLike
    _norm_buffer: List[ArrayLike]
    _estimation_algorithm: EstimationAlgorithm
    _monitor_window: int = 200 # TODO: Temporary internal attribute, remove when ready

    async def init_device(self) -> None:
        await super().init_device()
        self._norm_buffer = []
        self._estimation_algorithm = EstimationAlgorithm.MT_SEGMENTATION
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

    def get_estm_offset(self) -> int:
        return self._estm_offset

    def set_estm_offset(self, estm_offset: int) -> None:
        self._estm_offset = estm_offset
        self.info_stream(
            "%s has set estimation offset interval to: %d projections, state %s",
            self.__class__.__name__, self._estm_offset, self.get_state()
        )

    def get_center_of_rotation(self) -> float:
        return self._center_of_rotation

    def set_center_of_rotation(self, new_value: float) -> None:
        self._center_of_rotation = new_value

    def set_marae(self, marae: Tuple[int, int, int, int, int]) -> None:
        self._marae = marae

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
    @command(
        dtype_in=(float,),
        doc_in="wait window, check window, decision threshold"
    )
    async def estimate_center_of_rotation(self, args: Tuple[float]) -> None:
        """Estimates the center of rotation"""
        wait_window, check_window, err_threshold = int(args[0]), int(args[1]), args[2] 
        estimate: Awaitable[[AsyncIterator[ArrayLike], int, int, float], None]
        if self._estimation_algorithm in [EstimationAlgorithm.MT_SEGMENTATION,
                                          EstimationAlgorithm.MT_HOUGH_TRANSFORM]:
            estimate = self._estimate_mt
        else:
            estimate = self._estimate_corr
        await self._process_stream(estimate(self._receiver.subscribe(),wait_window=wait_window,
                                            check_window=check_window, err_threshold=err_threshold))

    @staticmethod
    def opt_func(angle_x: np.float64, center_p1: np.float64, radius_p2: np.float64,
                 phase_p3: np.float64) -> np.float64:
        """Defines the model function for nonlinear polynomial fit"""
        return center_p1 + radius_p2 * np.cos(angle_x + phase_p3)

    def _extract_marker_centroids(self, projection: ArrayLike) -> ArrayLike:
        """Implements the marker tracking strategies"""
        # NOTE: We crop a patch from the projection to localize the markers of interest so that the
        # algorithm can run efficiently. However, this has implications, especially for the left
        # side cropping. We need to add the same to estimated rotation axis as a correction factor.
        crop_left, crop_right, crop_vertical, num_markers, marker_diameter_px = self._marae
        self.info_stream("%s:meta information for rotation axis estimation",
                         self.__class__.__name__)
        self.info_stream("crop:[:%d, %d:%d], num_markers: %d, marker_diameter_px: %d",
                         crop_vertical, crop_left, crop_right, num_markers, marker_diameter_px)
        patch: ArrayLike = projection[:crop_vertical, crop_left: crop_right]
        centroids: List[ArrayLike] = []
        if self._estimation_algorithm == EstimationAlgorithm.MT_HOUGH_TRANSFORM:
            self.info_stream("%s:using Hough circle transform to track markers",
                             self.__class__.__name__)
            edges: ArrayLike = smf.canny(patch)
            # Hough circle transform fails to detect all markers upon using fixed radius, because
            # in practice markers don't have the same diameter. Hence, we need to define a range
            # to be resilient against heterogeneous marker sizes.
            hough_radii = np.arange((marker_diameter_px // 2)-1, (marker_diameter_px // 2)+1, 1)
            circ_refs: ArrayLike = smt.hough_circle(edges, hough_radii)
            _, coords_x, coords_y, _ = smt.hough_circle_peaks(circ_refs, hough_radii,
                                                              total_num_peaks=num_markers)
            centroids = [np.array([center_y, center_x]) for center_y, center_x in zip(
                coords_y, coords_x)]
        elif self._estimation_algorithm == EstimationAlgorithm.MT_SEGMENTATION:
            self.info_stream("%s:using segmentation to track markers", self.__class__.__name__)
            # Due to presence of many local impulses we cannot rely on single global threhsold to
            # separate the markers from everything else. Hence, we use stepwise thresholds. Since,
            # we are working with absorption projection, all values resulting from some sort of
            # absorption has higher responses compared to areas where minimum to no absorption takes
            # place. We take out those values with `global_threshold` and then we try to find out a
            # second threshold out of the intensities resulting from absorption which is
            # `local_threshold`. This time we are intrested in isolating the highest intensity
            # responses out of all absorption intensities. Those should belong to markers because
            # ideally they would have absorbed the most. At this point we use local_threshold to ask
            # from the entire patch for everything that is lesser than these extreme absorption
            # intensity responses. Since local_threshold distinguishes highest absorption intensity
            # responses from everything else, we would have markers in black(0) and everything else
            # in white(1). Hence, for labelling we need to use inverse of this mask.
            global_threshold = skf.threshold_otsu(patch)
            global_mask = patch[patch > global_threshold]
            local_threshold = skf.threshold_otsu(global_mask)
            local_mask = patch < local_threshold
            labels: ArrayLike = sms.label(~local_mask)
            regions: List[RegionProperties] = sms.regionprops(label_image=labels)
            centroids = [region.centroid for region in regions]
        # We sort marker centroids w.r.t vertical coordinate to be consistent with the markers
        # during curve_fit.
        centroids = sorted(centroids, key=lambda centroid: centroid[0])
        # NOTE: We are assuming that we have detected all the markers correctly. However, in
        # practice this is an edge case. Algorithm may fail to detect all the markers in complex
        # circumstances.
        assert(len(centroids) >= num_markers)
        if len(centroids) > num_markers:
            centroids = centroids[:num_markers]
        # To reduce memory foot print we store only the horizontal coordinates.
        return np.array(centroids)[:, 1]

    async def _estimate_mt(self, producer: AsyncIterator[ArrayLike],
                                           wait_window: int, check_window: int,
                                           err_threshold: float) -> None:
        """
        Estimates the center of rotation with marker tracking and nonlinear polynomial fit.
        """
        projection_count = 0
        marker_centroids: List[ArrayLike] = []
        estm_rot_axis: List[float] = []
        event_triggered = False
        crop_left, _, _, num_markers, _ = self._marae
        ffc = FlatCorrect(dark=self._dark, flat=self._flat, absorptivity=True)
        async for projection in ffc(producer):
            centroids_x: ArrayLike = self._extract_marker_centroids(projection=projection)
            marker_centroids.append(centroids_x)
            # We start curve-fit after initial wait window is passed. This is required for the
            # optimization method to be able to yield some values for the parameters. This wait
            # window is experimental and we take a conservative approach to choose one to avoid
            # RuntimeError during curve-fit.
            if projection_count < wait_window:
                self.info_stream(
                    "%s: Skipping estimation until [%d/%d]",
                    self.__class__.__name__,
                    projection_count + 1,
                    wait_window
                )
            else:
                # We start curve-fit after wait window is passed, however we do that with a
                # configurable offset of projections to gain momentum.
                if projection_count % self._estm_offset == 0:
                    try:
                        rot_axes:  List[np.float64] = []
                        for mix in range(num_markers):
                            x: ArrayLike = self._angle_dist[:projection_count + 1]
                            y: ArrayLike = np.array(marker_centroids)[:, mix]
                            params, _ = sop.curve_fit(f=self.opt_func,
                                                      xdata=np.squeeze(x),
                                                      ydata=np.squeeze(y))
                            # params[0] from curve-fit is the estimated axis of rotation w.r.t each
                            # marker. It depends upon the order of optimizable parameters, which
                            # is used to define the optimization function.
                            rot_axes.append(params[0])
                        self.info_stream(
                            "%s: Estimated rotation axis with projections [%d/%d]: %f",
                            self.__class__.__name__,
                            projection_count + 1,
                            self._num_radios,
                            np.median(rot_axes)
                        )
                        # We accumulate the median of the rotation axes with respect to all markers
                        # as our estimated target value. We observe the change in this value w.r.t
                        # a configurable error threshold over a time frame in terms of number of
                        # projections, given by check window.
                        # The _monitor_window property is used only for logging purpose. Ideally,
                        # we would remove it upon reaching a stable implementation.
                        estm_rot_axis.append(np.median(rot_axes))
                        if projection_count % self._monitor_window == 0:
                            self.info_stream(
                                    "Estimated gradient:\n%s\nEstimated difference:\n%s",
                                    np.round(np.abs(np.gradient(estm_rot_axis)), decimals=4),
                                    np.round(np.abs(np.gradient(estm_rot_axis) - err_threshold),
                                             decimals=1))
                        if len(estm_rot_axis) > check_window and not event_triggered:
                            # We compute gradient of the estimated centers of rotation and take the
                            # rounded value of the difference between gradient and error threshold.
                            # If the rounded difference is close to 0. for the duration stipulated
                            # by check window then we have landed on a stable value for the center
                            # of rotation.
                            grad_diff: ArrayLike = np.round(np.abs(
                                np.gradient(estm_rot_axis[-check_window:]) - err_threshold),
                                                            decimals=0)
                            if np.allclose(grad_diff, 0.):
                                # We set the last estimated value as the final center of rotation
                                # along with the correction for left side cropping.
                                self._center_of_rotation = estm_rot_axis[-1] + crop_left
                                self.info_stream("%s:estimated center of rotation %f, terminating.",
                                                 self.__class__.__name__,
                                                 self._center_of_rotation)
                                self.push_event("center_of_rotation", [], [],
                                                       self._center_of_rotation)
                                event_triggered = True
                                # At this point we assume to have estimated a stable value for the
                                # center of rotation, hence we can terminate the iteration to save
                                # compute resources.
                                break
                    except RuntimeError:
                        self.info_stream(
                            "%s could not find optimal parameters with projection: [%d/%d]",
                            self.__class__.__name__,
                            self._point_in_time - self._norm_window,
                            self._num_radios
                        )
            projection_count += 1

    async def _estimate_corr(self, producer: AsyncIterator[ArrayLike], wait_window: int,
                               check_window: int, err_threshold: float) -> None:
        """Estimates the center of rotation with correlation and gradient descent optimization"""
        raise NotImplementedError


if __name__ == "__main__":
    pass
