"""
rae.py
-----
Implements a device server to execute rotation axis estimation routines during acquisition.
"""
from typing import List, AsyncIterator
import numpy as np
import scipy.optimize as sop
from tango import DebugIt
from tango.server import attribute, command, AttrWriteType
from concert.ext.tangoservers.base import TangoRemoteProcessing
from concert.ext.ufo import FlatCorrect
from concert.imageprocessing import get_sphere_absorption_pattern, get_sphere_center_corr
from concert.typing import ArrayLike


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
        dtype=(int,),
        max_dim_x=2,
        access=AttrWriteType.WRITE,
        fset="set_attr_estm",
        doc="encapsulates attributes for axis estimation.e., init_wait, conv_window"
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
        self.info_stream("%s: acquisition attributes set to: %s", self.__class__.__name__,
                         str(self._attr_acq))

    def get_axis_of_rotation(self) -> float:
        return self._axis_of_rotation

    def set_axis_of_rotation(self, new_value: float) -> None:
        self._axis_of_rotation = new_value

    def set_attr_track(self, at: ArrayLike) -> None:
        self._attr_track = at
        self.info_stream("%s: tracking attributes set to: %s", self.__class__.__name__,
                         str(self._attr_track))

    def set_attr_estm(self, ae: ArrayLike) -> None:
        self._attr_estm = ae
        self.info_stream("%s: estimation attributes set to: %s", self.__class__.__name__,
                         str(self._attr_estm))

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
        init_wait, conv_window = self._attr_estm
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
                # NOTE: We replace the generic absorption pattern with a patch from the sphere
                # tracking on first iteration. We don't want to repeat this on the subsequent
                # iterations because exhaustive tests show that tracking error accumulates over time
                # when we change the pattern to be tracked repeatedly. Selecting one specific
                # pattern helps to stabilize the tracking.
                if proj_count == 0:
                    self._sphere = proj[yc - radius:yc + radius + 1,
                                    (xc + crop_left_px) - radius:(xc + crop_left_px) + radius + 1]
                ####################################################################################
                # TODO: Debugging code to visualize the sphere tracking. To be the removed before
                # merge.
                # if proj_count % 4 == 0:
                #     import matplotlib.pyplot as plt
                #     plot = proj[yc -radius:yc+radius+1,
                #                         (xc + crop_left_px)-radius:(xc + crop_left_px)+radius+1]
                #     plt.imshow(plot, cmap="gray")
                #     plt.show()
                ####################################################################################
                if proj_count > init_wait:
                    x: ArrayLike = self._angles[:proj_count]
                    y: ArrayLike = np.array(centers)[:proj_count, 1]
                    params, _ = sop.curve_fit(f=self._opt_func, xdata=np.squeeze(x),
                                              ydata=np.squeeze(y))
                    est_axis: float = crop_left_px + params[0]
                    est_axes.append(np.round(est_axis, decimals=1))
                    self.info_stream("%s: current estimation: %.1f", self.__class__.__name__,
                                     est_axes[-1])
                if not converged and len(est_axes) > conv_window:
                    if proj_count % 10 == 0:
                        self.debug_stream(
                                "estimation diff: %s",
                                np.abs(np.diff(np.trunc(est_axes[-conv_window:]))))
                    # To check for convergence we take the last `conv_window` estimates and check
                    # the difference in the integral part of the estimates. We want the integral
                    # part to remain stable throughout the last `conv_window` estimates.
                    if np.count_nonzero(np.abs(np.diff(np.trunc(est_axes[-conv_window:])))) == 0:
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
            # Unblock reco device server if the axis estimation routine is not successful for some
            # reason. In that case we take half of the projection width as a generic value.
            self.info_stream("%s: runtime error: %s, unblocking reco with generic value",
                             self.__class__.__name__, str(e))
            self._axis_of_rotation = proj.shape[1] / 2
            self.push_change_event("axis_of_rotation", self._axis_of_rotation)
        # Reset axis_of_rotation attribute for the subsequent acquisition.
        self._axis_of_rotation = None


if __name__ == "__main__":
    pass
