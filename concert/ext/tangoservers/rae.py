"""
rae.py
-----
Implements a device server to execute rotation axis estimation routines during acquisition.
"""
from enum import IntEnum
from typing import List, AsyncIterator, Tuple
import warnings
import numpy as np
import numpy.fft as nft
try:
    from numpy.typing import ArrayLike
except ModuleNotFoundError:
    from numpy import ndarray as ArrayLike
import scipy.ndimage as snd
import scipy.optimize as sop
import skimage.filters as skf
import skimage.measure as sms
import skimage.feature as smf
from skimage.measure._regionprops import RegionProperties
from tango import DebugIt, DevState
from tango.server import attribute, command, AttrWriteType
from concert.ext.tangoservers.base import TangoRemoteProcessing
from concert.ext.ufo import FlatCorrect
from concert.measures import rotation_axis


class Algorithm(IntEnum):
    """
    Feature flag for algorithm to use.
    We can either track markers as standalone approach or we can rely on a pre-computed sinogram
    to estimate deviation by motion estimation.
    """

    MARKER_TRACKING = 0
    MOTION_ESTIMATION = 1


class Tracking(IntEnum):
    """
    Feature flag for tracking using markers.
    We can choose to base our estimation statically on a specific marker (typically this is the one
    with largest displacement). Alternatively, we can determine `optimal` marker dynamically using
    an evaluation curve-fit after waiting period and MSE metric. As a third option we can enforce
    the combination of both markers.
    """
    
    MEAN_STATIC = 0  # Track both markers individually and take the mean of the estimation
    SINGLE_MARKER_STATIC = 1  # Estimate from a given single marker
    MEAN_MSE = 2  # Determine whether to take the mean or choose and optimal marker
    SINGLE_MARKER_MSE = 3  # Determine whether to take the mean or use a given single marker
    

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
        dtype=float,
        access=AttrWriteType.READ_WRITE,
        fget="get_axis_of_rotation",
        fset="set_axis_of_rotation",
        doc="axis of rotation to be estimated from incoming projections"
    )

    attr_track = attribute(
        label="Meta attributes for marker tracking",
        dtype=(int,),
        max_dim_x=5,
        access=AttrWriteType.WRITE,
        fset="set_attr_track",
        doc="encapsulates meta attributes for marker tracking i.e. \
        vert_crop, \
        crop_left_px, \
        crop_right_px, \
        radius, \
        use_marker"
    )

    attr_estm = attribute(
        label="Meta attributes for axis estimation",
        dtype=(float,),
        max_dim_x=6,
        access=AttrWriteType.WRITE,
        fset="set_attr_estm",
        doc="encapsulates attributes for axis estimationi.e., \
        offset, \
        mse_thresh, \
        init_wait, \
        avg_beta, \
        diff_thresh, \
        conv_window"
    )

    attr_mot_estm = attribute(
        label="Meta attributes for motion estimation",
        dtype=(int,),
        max_dim_x=2,
        access=AttrWriteType.WRITE,
        fset="set_attr_mot_estm",
        doc="encapsulates meta attributes for motion estimation i.e., \
        det_row_idx, \
        num_proj_corr"
    )

    _angles: ArrayLike
    _algorithm: Algorithm
    _tracking: Tracking

    async def init_device(self) -> None:
        await super().init_device()
        self._algorithm = Algorithm.MARKER_TRACKING
        self._tracking = Tracking.SINGLE_MARKER_MSE
        self._angles = np.array([])
        self.set_state(DevState.STANDBY)
        self._axis_of_rotation = -1
        self.set_change_event("axis_of_rotation", True, False)
        self.info_stream("%s initialized device with state: %s", self.__class__.__name__,\
                self.get_state())

    @attribute
    def algorithm(self) -> Algorithm:
        return self._algorithm

    @algorithm.setter
    def algorithm(self, algo: int) -> None:
        self._algorithm = Algorithm(algo)
        self.info_stream("%s: using algorithm %s", self.__class__.__name__, self._algorithm.name)

    @attribute
    def tracking(self) -> Tracking:
        return self._tracking

    @tracking.setter
    def tracking(self, trck: int) -> None:
        self._tracking = Tracking(trck)
        self.info_stream("%s: using tracking %s", self.__class__.__name__, self._tracking.name)

    def get_rot_angle(self) -> float:
        return self._rot_angle

    def set_rot_angle(self, rot_angle: float) -> None:
        self._rot_angle = rot_angle
        self.info_stream(
            "%s has set angle of rotation to: %f", self.__class__.__name__, self._rot_angle)

    def get_num_darks(self) -> int:
        return self._num_darks

    def set_num_darks(self, num_darks: int) -> int:
        self._num_darks = num_darks
        self.info_stream(
            "%s has set number of dark fields to: %d", self.__class__.__name__, self._num_darks)

    def get_num_flats(self) -> int:
        return self._num_flats

    def set_num_flats(self, num_flats: int) -> None:
        self._num_flats = num_flats
        self.info_stream(
            "%s has set number of flat fields to: %d", self.__class__.__name__, self._num_flats)

    def get_num_radios(self) -> int:
        return self._num_radios

    def set_num_radios(self, num_radios: int) -> None:
        self._num_radios = num_radios
        self.info_stream(
            "%s has set number of projections to: %d", self.__class__.__name__, self._num_radios)

    def get_axis_of_rotation(self) -> float:
        return self._axis_of_rotation

    def set_axis_of_rotation(self, new_value: float) -> None:
        self._axis_of_rotation = new_value

    def set_attr_track(self, at: ArrayLike) -> None:
        self._attr_track = at

    def set_attr_estm(self, ae: ArrayLike) -> None:
        self._attr_estm = ae

    def set_attr_mot_estm(self, ame: ArrayLike) -> None:
        self._attr_mot_estm = ame

    @DebugIt()
    @command()
    async def prepare_angular_distribution(self) -> None:
        """Prepares projected angular distribution as input values for nonlinear polynomial fit"""
        self._angles = np.linspace(0., self._rot_angle, self._num_radios)
        self.info_stream("%s: prepared angular distribution", self.__class__.__name__)

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
                "%s: processed %d %s projections", self.__class__.__name__, num_proj, name[1:])

    @DebugIt()
    @command()
    async def update_darks(self) -> None:
        await self._process_stream(self._process_flat_fields("_dark", self._receiver.subscribe()))

    @DebugIt()
    @command()
    async def update_flats(self) -> None:
        await self._process_stream(self._process_flat_fields("_flat", self._receiver.subscribe()))

    @DebugIt()
    @command()
    async def estimate_axis_of_rotation(self) -> None:
        """Estimates the center of rotation"""
        if self._algorithm == Algorithm.MARKER_TRACKING:
            await self._process_stream(self._track_markers(self._receiver.subscribe()))
        elif self._algorithm == Algorithm.MOTION_ESTIMATION:
            await self._process_stream(self._estimate_motion(self._receiver.subscribe()))
        else:
            raise RuntimeError(f"unsupported method: {self._algorithm}")

    @staticmethod
    def _opt_func(angle_x: float, center_p1: float, radius_p2: float, phase_p3: float) -> float:
        """Defines the optimization function"""
        return center_p1 + radius_p2 * np.cos(angle_x + phase_p3)

    @staticmethod
    def _eval_func(angle_x: float, amplitude: float, phase: float, offset: float) -> float:
        """
        Defines the evaluation function (same as optimization function having the parameters
        interpreted differently)
        """
        return amplitude * np.cos(angle_x + phase) + offset

    @staticmethod
    def _find_centroids_corr(patch: ArrayLike, radius: int = 8,
                                px_offset: int = 4) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        Extracts marker centroids from a patch of the projection using normalized cross
        correlation with a simulated absorption pattern of a metal sphere having given radius. A
        pixel offset is used to extract two square patches keeping the markers at the center. These
        patches are used to validate with some validation criteria that the centroids are extracted
        correctly.

        :param patch: patch from the projection to locate markers.
        :type patch: ArrayLike
        :param radius: approxmate radius of the markers in pixels.
        :type radius: int
        :param px_offset: pixel offset from marker center to define marker patches.
        :type px_offset: int
        :return: extracted marker centroids and respective patches for validation.
        :rtype: Tuple[ArrayLike, ArrayLike, ArrayLike]
        """
        # Define a generic absorption pattern of a metal spehere with given radius
        y, x = np.mgrid[-radius:radius+1, -radius:radius+1]
        mask = np.where(radius ** 2 - x ** 2 - y ** 2 >= 0)
        sphere = np.zeros((2 * radius + 1, 2 * radius + 1))
        sphere[mask] = 2 * np.sqrt(radius ** 2 - x[mask] ** 2 - y[mask] ** 2)
        # Correlate generic absorption pattern with the patch from the projection
        corr: ArrayLike = nft.ifft2(nft.fft2(patch - patch.mean()) * np.conjugate(
            nft.fft2(sphere - sphere.mean(), s=patch.shape))).real
        ym_1, xm_1 = np.unravel_index(np.argmax(corr), corr.shape)
        # Create a mask to obscure already recorded response to detect the next response using
        # argmax. To safeguard against the irregularity of the marker shape we take 3 times of the
        # marker radius to define the mask.
        msy, msx = np.mgrid[:corr.shape[0], :corr.shape[1]]
        corr[(msy - ym_1)**2 + (msx - xm_1)**2 <= 3*radius**2] = 0
        ym_2, xm_2 = np.unravel_index(np.argmax(corr), corr.shape)
        # TODO: Vaidate the following approach of adding the radius to determine the offset. This
        # might be a source of estimation error.
        ym_1 += radius
        xm_1 += radius
        ym_2 += radius
        xm_2 += radius
        centroids: List[List[int]] = [[ym_1, xm_1], [ym_2, xm_2]]
        centroids: ArrayLike = np.array(sorted(centroids, key=lambda centroid: centroid[0]))
        # Define dimensions of the marker patch w.r.t. marker centroids
        patch_dim = radius + px_offset
        mp1: ArrayLike = patch[centroids[0,0]-patch_dim:centroids[0,0]+patch_dim,
                               centroids[0,1]-patch_dim:centroids[0,1]+patch_dim]
        mp2: ArrayLike = patch[centroids[1,0]-patch_dim:centroids[1,0]+patch_dim,
                               centroids[1,1]-patch_dim:centroids[1,1]+patch_dim]
        return centroids, mp1, mp2

    @staticmethod
    def _circularity_of(region: RegionProperties) -> float:
        """
        Defines a circularity metric on top of region properties, which can be used to filter
        the regions. Since we know that the absorption pattern of the markers are approximately
        circular in the projection, the metric (4 * pi * pi * r^2) / (2 * pi * r * 2 * pi * r)
        should yield a value in the range (0, 1) for an approximate circular region.

        :param region: region properties
        :type region: skimage.measure._regionprops.RegionProperties
        :return: circularity value for a given region.
        :rtype: float
        """
        return (4 * np.pi * region.area)/(region.perimeter * region.perimeter)

    def _is_valid(self, patch: ArrayLike, circ: float = 0.8) -> bool:
        """
        Determines the validity of a marker patch against shape and given circularity value.

        :param patch: patch which should be square and should contain a marker.
        :type patch: ArrayLike
        :param circ: circularity threshold value to determine if the patch contains a marker
        :type circ: float
        :return: if patch is square and constains a marker.
        :rtype: bool
        """
        if patch.shape[0] != patch.shape[1]:
            return False
        threshold: float = skf.threshold_otsu(patch)
        mask: ArrayLike = patch < threshold
        labels: ArrayLike = sms.label(~mask)
        region: RegionProperties = sorted(
            sms.regionprops(label_image=labels), key=lambda r: r.area_filled, reverse=True)[0]
        if self._circularity_of(region=region) < circ:
            return False
        return True

    def _find_centroids_otsu(self, strip: ArrayLike,
                             local_thresh_iter: int = 1) -> Tuple[float, float]:
        """
        Extracts marker centroids from localized horizontal strip of the projection containing the
        markers. For this we use Otsu's method of threshold-based segmentation. We can choose to
        apply threshold a number of times (if required).

        :param strip: strip from the projection containing the marker.
        :type strip: ArrayLike
        :param local_thresh_iter: number of times to apply the threshold
        :type local_thresh_iter: int
        :return: local marker centroids extracted from the strip.
        :rtype: Tuple[float, float]
        """
        threshold: float = skf.threshold_otsu(strip)
        mask: ArrayLike = strip[strip > threshold]
        for _ in range(local_thresh_iter):
            threshold = skf.threshold_otsu(mask)
            mask = strip[strip > threshold]
        threshold = skf.threshold_otsu(mask)
        final_mask: ArrayLike = strip < threshold
        labels: ArrayLike = sms.label(~final_mask)
        regions: List[RegionProperties] = sms.regionprops(label_image=labels)
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore")
            regions = list(filter(
                lambda region: self._circularity_of(region=region) not in [-np.inf, np.inf],
                regions))
        regions = sorted(regions, key=lambda r: r.area_filled, reverse=True)
        return regions[0].centroid

    async def _track_markers(self, producer: AsyncIterator[ArrayLike]) -> None:
        """
        Estimates the center of rotation with marker tracking and nonlinear polynomial fit.

        :param: producer: asynchronous generator of projections.
        :type producer: AsyncIterator[ArrayLike]
        """
        vert_crop, crop_left_px, crop_right_px, radius, use_marker = self._attr_track
        offset, mse_thresh, init_wait, avg_beta, diff_thresh, conv_window = self._attr_estm
        offset, init_wait, conv_window = int(offset), int(init_wait), int(conv_window)
        proj_count = 0
        # Optimal marker value would be set during evaluation after initial wait window.
        optimal_marker = -1
        centroids_x: List[List[float]] = []
        ffc = FlatCorrect(dark=self._dark, flat=self._flat, absorptivity=True)
        est_axes: List[float] = []
        evaluated: bool = False
        converged: bool = False
        try:
            async for projection in ffc(producer):
                # Crop projection to reduce tracking complexity.
                height: int = projection.shape[0] // vert_crop
                crop_right: int = projection.shape[1] - crop_right_px
                if height > 0:
                    patch: ArrayLike = projection[:height, crop_left_px:crop_right]
                else:
                    patch: ArrayLike = projection[height:, crop_left_px:crop_right]
                # Tracking: use normalized cross correlation to localize the markers on first
                # projection and use segmentation on individual marker strips thereafter.
                if proj_count == 0:
                    cnt, mp1, mp2 = self._find_centroids_corr(patch=patch)
                    if not self._is_valid(patch=mp1) or not self._is_valid(patch=mp2):
                        cnt, mp1, mp2 = self._find_centroids_corr(patch=skf.laplace(skf.gaussian(
                            patch, sigma=4)))
                    centroids_x.append([cnt[0,1], cnt[1,1]])
                else:
                    # Define two marker strips with two pixels of offset on top of radius
                    strip1: ArrayLike = patch[cnt[0,0]-(radius+2):cnt[0,0]+(radius+2), :]
                    strip2: ArrayLike = patch[cnt[1,0]-(radius+2):cnt[1,0]+(radius+2), :]
                    _, mx1 = self._find_centroids_otsu(strip=strip1)
                    _, mx2 = self._find_centroids_otsu(strip=strip2)
                    centroids_x.append([mx1, mx2])
                if proj_count > init_wait:
                    # Evaluation: compare tracking quality of the markers based on provided criteria or
                    # decide to use a single static marker for estimation. Evaluation happens a single
                    # time and result would be used thereafter.
                    if not evaluated:
                        if self._tracking in [Tracking.MEAN_MSE, Tracking.SINGLE_MARKER_MSE]:
                            # Compute MSE for both markers using evaluation function.
                            ang_x: ArrayLike = self._angles[:proj_count]
                            m1_cnt: ArrayLike = np.array(centroids_x)[:proj_count, 0]
                            m2_cnt: ArrayLike = np.array(centroids_x)[:proj_count, 1]
                            eval_params_m1, _ = sop.curve_fit(f=self._eval_func, xdata=ang_x,
                                                              ydata=np.squeeze(m1_cnt))
                            eval_params_m2, _ = sop.curve_fit(f=self._eval_func, xdata=ang_x,
                                                              ydata=np.squeeze(m2_cnt))
                            m1_pred: ArrayLike = self._eval_func(ang_x, *eval_params_m1)
                            m2_pred: ArrayLike = self._eval_func(ang_x, *eval_params_m2)
                            m1_mse: float = np.mean((m1_cnt - m1_pred)**2)
                            m2_mse: float = np.mean((m2_cnt - m2_pred)**2)
                            # Is both cases if abs diff of MSE values remain within threshold, it means
                            # that tracking quality of both markers are identical to each other. We can
                            # take the mean. Since optimal_marker is initialized as -1 we do nothing.
                            if self._tracking == Tracking.MEAN_MSE:
                                if np.abs(m1_mse - m2_mse) > mse_thresh:
                                    optimal_marker = np.argmin([m1_mse, m2_mse])
                                    self.info_stream("%s: (using MSE) chosen marker: %d",
                                                     self.__class__.__name__, optimal_marker)
                            elif self._tracking == Tracking.SINGLE_MARKER_MSE:
                                if np.abs(m1_mse - m2_mse) > mse_thresh:
                                    # use_marker must have a valid marker index when tracking method is
                                    # chosen as Tracking.SINGLE_MARKER_*
                                    assert(use_marker in [0, 1])
                                    optimal_marker = use_marker
                                    self.info_stream("%s: (using MSE) selected marker: %d",
                                                     self.__class__.__name__, optimal_marker)
                        else:
                            if self._tracking == Tracking.MEAN_STATIC:
                                # Do nothing as the optimal marker is initialized to -1, which means
                                # estimation mean would be used.
                                pass
                            elif self._tracking == Tracking.SINGLE_MARKER_STATIC:
                                # Set optimal marker value by the provided use_marker configuration,
                                # either 0 or 1. We must have a valid marker index as use_marker when
                                # tracking method is chosen as Tracking.SINGLE_MARKER_*
                                assert (use_marker in [0, 1])
                                optimal_marker = use_marker
                        if optimal_marker == -1:
                            self.info_stream("%s: estimations from both markers will be combined",
                                             self.__class__.__name__)
                        evaluated = True
                    # Estimation: perform curve-fit to estimate the axis of rotation. Estimation happens
                    # either for each projection or with an offset. Based on the provided configuration
                    # and evaluation estimation is performed on a single marker or mean from both.
                    if proj_count % offset == 0:
                        x: ArrayLike = self._angles[:proj_count:offset]
                        if optimal_marker != -1:
                            y_m: ArrayLike = np.array(centroids_x)[:proj_count:offset, optimal_marker]
                            params, _ = sop.curve_fit(f=self._opt_func, xdata=np.squeeze(x),
                                                      ydata=np.squeeze(y_m))
                            est_axis: float = crop_left_px + params[0]
                        else:
                            y_m1: ArrayLike = np.array(centroids_x)[:proj_count:offset, 0]
                            y_m2: ArrayLike = np.array(centroids_x)[:proj_count:offset, 1]
                            params_m1, _ = sop.curve_fit(f=self._opt_func, xdata=np.squeeze(x),
                                                         ydata=np.squeeze(y_m1))
                            params_m2, _ = sop.curve_fit(f=self._opt_func, xdata=np.squeeze(x),
                                                         ydata=np.squeeze(y_m2))
                            est_axis: float = crop_left_px + np.mean([params_m1[0], params_m2[0]])
                        if len(est_axes) == 0:
                            est_axes.append(np.round(est_axis, decimals=3))
                        else:
                            avg_est: float = (avg_beta * est_axes[-1]) + ((1 - avg_beta) * est_axis)
                            avg_est /= (1 - avg_beta**proj_count)
                            est_axes.append(np.round(avg_est, decimals=3))
                    # Convergence: check if we have landed on a stable value. This check happens for
                    # each projection after a number of past values were estimated, defined by the
                    # evaluation window.
                    if not converged:
                        if len(est_axes) > conv_window:
                            # TODO: Debug log, remove when ready
                            if proj_count % conv_window == 0:
                                self.info_stream("%s estimated axis: %.3f", self.__class__.__name__,
                                                 est_axes[-1])
                            if np.all(np.round(np.abs(np.diff(est_axes[-conv_window:])),
                                               decimals=3) <= diff_thresh):
                                # Attribute axis_of_rotation encapsulates axis_angle_y(roll) and
                                # axis_angle_x(tilt) angles, which are not in effect at this point
                                # int time.
                                self._axis_of_rotation = est_axes[-1]
                                self.info_stream("%s: converged at: %s", self.__class__.__name__,
                                                 self._axis_of_rotation)
                                self.push_change_event("axis_of_rotation", self._axis_of_rotation)
                                converged = True
                proj_count += 1
            # If not converged final estimate is used to carry out the reconstruction.
            if not converged:
                self._axis_of_rotation = est_axes[-1]
                self.info_stream("%s: estimated: %s", self.__class__.__name__, self._axis_of_rotation)
                self.push_change_event("axis_of_rotation", self._axis_of_rotation)
        except Exception as e:
            self.info_stream("%s encountered runtime error: %s, unblocking reco with generic value",
                             self.__class__.__name__, str(e))
            # Unblock reco device server with a generic value
            self._axis_of_rotation = 0.
            self.push_change_event("axis_of_rotation", self._axis_of_rotation)

    async def _estimate_motion(self, producer: AsyncIterator[ArrayLike]) -> None:
        """Estimates center of rotation with correlation with phase correlation"""
        # We assert, that there is a pre-computed known value exists for the center of rotation
        # because this method estimates the potential error for the same.
        assert(np.count_nonzero(self._axis_of_rotation) != 0)
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
        self._axis_of_rotation += corr_peak_loc[1]
        self.info_stream("%s: estimated axis error %d pixles, revised center of rotation: %d",
                         self.__class__.__name__, axis_correction, self._axis_of_rotation)
        self.push_change_event("axis_of_rotation", self._axis_of_rotation)


if __name__ == "__main__":
    pass
