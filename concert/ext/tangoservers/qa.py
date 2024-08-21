"""
qa.py
-----
Implements a device server to execute quality assurance routines during acquisition.
"""
from typing import List, AsyncIterator, Tuple
import numpy as np
from numpy.typing import ArrayLike
import scipy.ndimage as snd
import scipy.optimize as sop
import skimage.filters as skf
import skimage.measure as sms
from skimage.measure._regionprops import RegionProperties
from tango import DebugIt, DevState, CmdArgType, EventType
from tango.server import attribute, command, AttrWriteType
from concert.ext.tangoservers.base import TangoRemoteProcessing
from concert.ext.ufo import FlatCorrect, GaussianBlur


class QualityAssurance(TangoRemoteProcessing):
    """
    Implements Tango device server to encapsulate quality assurance routines.
    """
    rot_angle = attribute(
        label="Rot_Angle",
        dtype=float,
        access=AttrWriteType.READ_WRITE,
        fget="get_rot_angle",
        fset="set_rot_angle",
        doc="overall angle of rotation for current acquisition"
    )

    num_darks = attribute(
        label="Num_Darks",
        dtype=int,
        access=AttrWriteType.READ_WRITE,
        fget="get_num_darks",
        fset="set_num_darks",
        doc="overall number of dark field acquisitions"
    )

    num_flats = attribute(
        label="Num_Flats",
        dtype=int,
        access=AttrWriteType.READ_WRITE,
        fget="get_num_flats",
        fset="set_num_flats",
        doc="overall number of flat field acquisitions"
    )

    num_radios = attribute(
        label="Num_Proj",
        dtype=int,
        access=AttrWriteType.READ_WRITE,
        fget="get_num_radios",
        fset="set_num_radios",
        doc="overall number of projections to be acquired"
    )

    estm_offset = attribute(
        label="Estm_Offset",
        dtype=int,
        access=AttrWriteType.READ_WRITE,
        fget="get_estm_offset",
        fset="set_estm_offset",
        doc="offset in number of projections to be used to run estimation"
    )

    center_of_rotation = attribute(
        label="Center_Of_Rotation",
        dtype=float,
        access=AttrWriteType.READ_WRITE,
        fget="get_center_of_rotation",
        fset="set_center_of_rotation",
        doc="estimated center of rotation"
    )

    _angle_dist: ArrayLike
    _norm_buffer: List[ArrayLike]
    _monitor_window: int  # TODO: Temporary internal attribute, remove when ready

    async def init_device(self) -> None:
        await super().init_device()
        self._norm_buffer = []
        self._monitor_window = 100
        self.info_stream("%s initialized device with state: %s",
                         self.__class__.__name__, self.get_state())

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
        doc_in="number of markers, wait window, check window, sigma, decision threshold"
    )
    async def estimate_center_of_rotation(self, args: Tuple[float]) -> None:
        """Estimates the center of rotation"""
        num_markers, wait_window, check_window = int(args[0]), int(args[1]), int(args[2])
        sigma, err_threshold = args[3], args[4]
        self.info_stream("%s starting estimation using markers=%d, wait_time=%d, sigma=%f",
                         self.__class__.__name__, num_markers, wait_window, sigma)
        await self._process_stream(self._estimate_from_markers(self._receiver.subscribe(),
                                                               num_markers=num_markers,
                                                               wait_window=wait_window,
                                                               check_window=check_window,
                                                               sigma=sigma,
                                                               err_threshold=err_threshold))

    @staticmethod
    def opt_func(angle_x: np.float64, center_p1: np.float64, radius_p2: np.float64,
                 phase_p3: np.float64) -> np.float64:
        """Defines the model function for nonlinear polynomial fit"""
        return center_p1 + radius_p2 * np.cos(angle_x + phase_p3)

    async def _estimate_from_markers(self, producer: AsyncIterator[ArrayLike], num_markers: int,
                                     wait_window: int, check_window: int, sigma: float,
                                     err_threshold: float) -> None:
        """
        Estimates the center of rotation with marker tracking and nonlinear polynomial fit
        """
        projection_count = 0
        marker_centroids: List[ArrayLike] = []
        estm_rot_axis: List[float] = []
        event_triggered = False
        num_intensity_classes = 3
        ffc = FlatCorrect(dark=self._dark, flat=self._flat, absorptivity=False)
        # Derive Gaussian kernel size from standard deviation with formula: 2 * ceil(3 * sigma) + 1.
        gb = GaussianBlur(kernel_size=2 * np.ceil(3 * sigma).astype(int) + 1, sigma=sigma)
        # Process all projections for potential marker centroids but run curve-fit conditionally.
        async for projection in gb(ffc(producer)):
            thresh: ArrayLike = skf.threshold_multiotsu(projection, classes=num_intensity_classes)
            mask: ArrayLike = projection < thresh.min()
            labels: ArrayLike = sms.label(mask)
            regions: List[RegionProperties] = sms.regionprops(label_image=labels)
            regions = sorted(regions, key=lambda region: region.area, reverse=True)
            # TODO: Here we implicitly assume that the detected number of regions would always be
            # greater than the expected number of markers. In practice, this is an edge case because
            # it might not always be true.
            if len(regions) > num_markers:
                regions = regions[:num_markers]
            marker_centroids.append(np.array(
                list(map(lambda region: region.centroid, regions))))
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
                        # TODO: This is a potential edge case. Can this be guaranteed that the
                        # segmentation algorithm would always be able to detect desired number of
                        # markers and give us their centroids ?
                        # OR, a better idea could be that we infer the detected number of markers
                        # from the shape of the accumulation array. That actually could introduce
                        # errors in estimation. This needs to be explored and experimented with.
                        # Ideal solution is the simplest one that works reasonably well all the
                        # time and not perfectly only some times.
                        for mix in range(num_markers):
                            x: ArrayLike = self._angle_dist[:projection_count + 1]
                            y: ArrayLike = np.array(marker_centroids)[:, mix, 1]
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
                                         decimals=1)
                            )
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
                                # We set the last estimated value as the final center of rotation.
                                self._center_of_rotation = estm_rot_axis[-1]
                                self.info_stream("%s: Estimated center of rotation %f, terminating.",
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


if __name__ == "__main__":
    pass
