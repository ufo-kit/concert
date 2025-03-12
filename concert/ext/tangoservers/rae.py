"""
rae.py
-----
Implements a device server to execute rotation axis estimation routines during acquisition.
"""
from typing import List, AsyncIterator
import numpy as np
try:
    from numpy.typing import ArrayLike
except ModuleNotFoundError:
    from numpy import ndarray as ArrayLike
import scipy.optimize as sop
from tango import DebugIt, DevState
from tango.server import attribute, command, AttrWriteType
from concert.ext.tangoservers.base import TangoRemoteProcessing
from concert.ext.ufo import FlatCorrect
from concert.imageprocessing import get_sphere_absorption_pattern, get_sphere_center_corr


class TangoRotationAxisEstimator(TangoRemoteProcessing):
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

    attr_acq = attribute(
        label="Meta attribute for acquisition",
        dtype=(int,),
        max_dim_x=3,
        access=AttrWriteType.READ_WRITE,
        fget="get_attr_acq",
        fset="set_attr_acq",
        doc="encapsulates acquisition meta information i.e., #darks, #flats, #radios"
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
        max_dim_x=4,
        access=AttrWriteType.WRITE,
        fset="set_attr_track",
        doc="encapsulates meta attributes for marker tracking i.e. \
        crop_vert_prop, \
        crop_left_px, \
        crop_right_px, \
        radius"
    )

    attr_estm = attribute(
        label="Meta attributes for axis estimation",
        dtype=(float,),
        max_dim_x=4,
        access=AttrWriteType.WRITE,
        fset="set_attr_estm",
        doc="encapsulates attributes for axis estimation.e., \
        init_wait, \
        avg_beta, \
        diff_thresh, \
        conv_window"
    )

    async def init_device(self) -> None:
        await super().init_device()
        # We initialize axis_of_rotation attribute with None as a safeguard against mis-fire of the
        # tango events. Reco device should only take a non-None value of axis of rotation into
        # account.
        self._axis_of_rotation = None
        self.set_change_event("axis_of_rotation", True, False)
        self.info_stream("%s initialized device with state: %s", self.__class__.__name__,
                         self.get_state())

    def get_rot_angle(self) -> float:
        return self._rot_angle

    def set_rot_angle(self, rot_angle: float) -> None:
        self._rot_angle = rot_angle
        self.info_stream(
            "%s has set angle of rotation to: %f", self.__class__.__name__, self._rot_angle)

    def get_attr_acq(self) -> ArrayLike:
        return self._attr_acq

    def set_attr_acq(self, aa: ArrayLike) -> None:
        self._attr_acq = aa

    def get_axis_of_rotation(self) -> float:
        return self._axis_of_rotation

    def set_axis_of_rotation(self, new_value: float) -> None:
        self._axis_of_rotation = new_value

    def set_attr_track(self, at: ArrayLike) -> None:
        self._attr_track = at

    def set_attr_estm(self, ae: ArrayLike) -> None:
        self._attr_estm = ae

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
        self.info_stream("%s: processed %d %s projections", self.__class__.__name__, num_proj,
                         name[1:])

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
        _, _, num_radios = self._attr_acq
        _, _, _, radius = self._attr_track
        self._angles = np.linspace(0., self._rot_angle, num_radios)
        self._sphere = get_sphere_absorption_pattern(radius=radius)
        self.info_stream("%s: starting estimation", self.__class__.__name__)
        await self._process_stream(self._track_spheres(self._receiver.subscribe()))

    @staticmethod
    def _opt_func(angle_x: float, center_p1: float, radius_p2: float, phase_p3: float) -> float:
        """Defines the optimization function"""
        return center_p1 + radius_p2 * np.cos(angle_x + phase_p3)

    async def _track_spheres(self, producer: AsyncIterator[ArrayLike]) -> None:
        """
        Estimates the center of rotation with marker tracking and nonlinear polynomial fit.

        :param: producer: asynchronous generator of projections.
        :type producer: AsyncIterator[ArrayLike]
        """
        crop_vert_prop, crop_left_px, crop_right_px, radius = self._attr_track
        init_wait, avg_beta, diff_thresh, conv_window = self._attr_estm
        init_wait, conv_window = int(init_wait), int(conv_window)
        proj_count = 0
        centers: List[List[int]] = []
        ffc = FlatCorrect(dark=self._dark, flat=self._flat, absorptivity=True)
        est_axes: List[float] = []
        converged: bool = False
        try:
            async for proj in ffc(producer):
                yc, xc = get_sphere_center_corr(proj=proj, sphere=self._sphere, radius=radius,
                                                crop_vert_prop=crop_vert_prop,
                                                crop_left_px=crop_left_px,
                                                crop_right_px=crop_right_px)
                centers.append([yc, xc])
                if proj_count > init_wait:
                    x: ArrayLike = self._angles[:proj_count]
                    y: ArrayLike = np.array(centers)[:proj_count, 1]
                    params, _ = sop.curve_fit(f=self._opt_func, xdata=np.squeeze(x),
                                              ydata=np.squeeze(y))
                    est_axis: float = crop_left_px + params[0]
                    if len(est_axes) == 0:
                        est_axes.append(np.round(est_axis, decimals=1))
                    else:
                        avg_est: float = (avg_beta * est_axes[-1]) + ((1 - avg_beta) * est_axis)
                        avg_est /= (1 - avg_beta**proj_count)
                        est_axes.append(np.round(avg_est, decimals=1))
                    self.info_stream("%s: current estimation: %.1f", self.__class__.__name__,
                                     est_axes[-1])
                # Convergence: check if we have landed on a stable value. This check happens for
                # each projection after given number of estimations are available. Since we want to
                # achieve less than half-pixel error in estimation, we round the estimated values
                # to 1 decimal position and compare against a threshold value, which should be less
                # than configured diff_thresh.
                if not converged and len(est_axes) > conv_window:
                    if proj_count % 10 == 0:
                        self.debug_stream(
                                "estimation diff: %s",
                                np.round(np.abs(np.diff(est_axes[-conv_window:])), decimals=1))
                    if np.all(np.round(np.abs(np.diff(est_axes[-conv_window:])),
                                       decimals=1) < diff_thresh):
                        self._axis_of_rotation = est_axes[-1]
                        self.info_stream("%s: converged at: %.1f", self.__class__.__name__,
                                         self._axis_of_rotation)
                        self.push_change_event("axis_of_rotation", self._axis_of_rotation)
                        converged = True
                        break
                proj_count += 1
            # If not converged final estimate is used to carry out the reconstruction.
            if not converged:
                self._axis_of_rotation = est_axes[-1]
                self.info_stream("%s: final estimate: %.1f", self.__class__.__name__,
                                 self._axis_of_rotation)
                self.push_change_event("axis_of_rotation", self._axis_of_rotation)
        except Exception as e:
            self.info_stream("%s: runtime error: %s, unblocking reco with generic value",
                             self.__class__.__name__, str(e))
            # Unblock reco device server if the axis estimation routine is not successful for some
            # reason. In that case we take half of the projection width as a generic value.
            self._axis_of_rotation = proj.shape[1] / 2
            self.push_change_event("axis_of_rotation", self._axis_of_rotation)
        # Set axis_of_rotation attribute back to None to safe-guard against mis-fire of tango
        # events.
        self._axis_of_rotation = None
        try:
            # NOTE: We need to put this strategy into test. Since this routine is supposed to be
            # executed in the reconstruction server its implication on the reconstruction workflow
            # needs to be understood.
            import cupy as cp
            cp._default_memory_pool.free_all_blocks()
            self.info_stream("%s: attempted to release unused GPU memory")
        except ModuleNotFoundError:
            # If cupy is available we ensure to release all allocated GPU memory for current
            # stream. We don't need to do anything specific otherwise.
            pass


if __name__ == "__main__":
    pass
