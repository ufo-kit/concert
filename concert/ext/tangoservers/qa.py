"""
qa.py
-----
Implements a device server to execute quality assurance routines during
acquisition.
"""
from typing import List, AsyncGenerator, Awaitable
import numpy as np
from numpy.typing import ArrayLike
import scipy.ndimage as snd
import scipy.optimize as sop
import skimage.filters as skf
import skimage.measure as sms
from skimage.measure._regionprops import RegionProperties
from tango import DebugIt, DevState, CmdArgType
from tango.server import attribute, command, AttrWriteType
from concert.ext.tangoservers.base import TangoRemoteProcessing


class QualityAssurance(TangoRemoteProcessing):
    """
    Implements Tango device server to encapsulate quality assurance
    routines
    """

    lpf_size = attribute(
            label="Low_Pass_Filter_Size",
        dtype=int,
        access=AttrWriteType.READ_WRITE,
        fget="get_lpf_size",
        fset="set_lpf_size",
        doc="low pass filter size to filter flatcorrected projections"
    )

    num_markers = attribute(
        label="Num_Markers",
        dtype=int,
        access=AttrWriteType.READ_WRITE,
        fget="get_num_markers",
        fset="set_num_markers",
        doc="number of markers to estimate axis of rotation from"
    )

    rot_angle = attribute(
        label="Rot_Angle",
        dtype=float,
        access=AttrWriteType.READ_WRITE,
        fget="get_rot_angle",
        fset="set_rot_angle",
        doc="overall angle of rotation for current acquisition"
    )

    num_proj = attribute(
        label="Num_Proj",
        dtype=int,
        access=AttrWriteType.READ_WRITE,
        fget="get_num_proj",
        fset="set_num_proj",
        doc="overall number of projections to be acquired"
    )

    wait_interval = attribute(
        label="Wait_Interval",
        dtype=int,
        access=AttrWriteType.READ_WRITE,
        fget="get_wait_interval",
        fset="set_wait_interval",
        doc="initial interval to wait before starting rotation axis estimation"
    )

    _marker_centroids: List[ArrayLike]
    _point_in_time: int
    _angle_dist: ArrayLike

    async def init_device(self) -> None:
        await super().init_device()
        self._marker_centroids = []
        self._point_in_time = 0
        self.info_stream("%s initialized device with state: %s",
                         self.__class__.__name__, self.get_state())

    def get_lpf_size(self) -> int:
        return self._lpf_size

    def set_lpf_size(self, lpf_size) -> None:
        self._lpf_size = lpf_size
        self.info_stream(
            "%s has set low pass filter size to: %d, state: %s",
            self.__class__.__name__, self.get_lpf_size(), self.get_state()
        )

    def get_num_markers(self) -> int:
        return self._num_markers

    def set_num_markers(self, num_markers) -> None:
        self._num_markers = num_markers
        self.info_stream(
            "%s has set number of markers to: %d, state: %s",
            self.__class__.__name__, self.get_num_markers(), self.get_state()
        )

    def get_rot_angle(self) -> float:
        return self._rot_angle

    def set_rot_angle(self, rot_angle: float) -> None:
        self._rot_angle = rot_angle
        self.info_stream(
            "%s has set overall angle of rotation to: %f, state: %s",
            self.__class__.__name__, self.get_rot_angle(), self.get_state()
        )

    def get_num_proj(self) -> int:
        return self._num_proj

    def set_num_proj(self, num_proj) -> None:
        self._num_proj = num_proj
        self.info_stream(
            "%s has set overall number of projections to: %d, state: %s",
            self.__class__.__name__, self.get_num_proj(), self.get_state()
        )

    def get_wait_interval(self) -> int:
        return self._wait_interval

    def set_wait_interval(self, wait_interval) -> None:
        self._wait_interval = wait_interval
        self.info_stream(
            "%s has set wait onterval for estimation to: %d projections, state: %s",
            self._class__.__name__, self.get_wait_interval(), self.get_state()
        )

    @DebugIt()
    @command()
    async def prepare_angular_distribution(self) -> None:
        self._angle_dist = np.linspace(0., self.get_rot_angle(),
                                       self.get_num_proj())

    @DebugIt()
    @command()
    async def derive_rot_axis(self) -> None:
        await self._process_stream(self._walker_consume_alike())

    @staticmethod
    def opt_func(angle_x: np.float64,
                 center_p1: np.float64,
                 radius_p2: np.float64,
                 phase_p3: np.float64) -> np.float64:
        return center_p1 + radius_p2 * np.cos(angle_x + phase_p3)

    async def _walker_consume_alike(self) -> None:
        # Executes _walker_write_alike coroutine
        await self._walker_write_alike(self._receiver.subscribe())

    async def _walker_write_alike(self, producer: AsyncGenerator[ArrayLike, None]) -> Awaitable:
        # Executes the coroutine returned by _walker_create_writer_alike
        return await self._walker_create_writer_alike(producer)

    def _walker_create_writer_alike(self, producer: AsyncGenerator[ArrayLike, None]) -> Awaitable:
        # Returns a coroutine without executing it
        return self._compute_axis(producer)

    async def _compute_axis(self, producer: AsyncGenerator[ArrayLike, None]) -> None:
        # TODO: Implement flat-correction utility. For now assume that projections
        # are flat-corrected
        async for projection in producer:
            self._point_in_time += 1
            projection = snd.median_filter(projection, size=self._lpf_size)
            thresh: ArrayLike = skf.threshold_multiotsu(projection, classes=3)
            mask: ArrayLike = projection < thresh.min()
            labels: ArrayLike = sms.label(mask)
            regions: List[RegionProperties] = sms.regionprops(label_image=labels)
            regions = sorted(regions, key=lambda region: region.area, reverse=True)
            if len(regions) > self.get_num_markers():
                regions = regions[:self.get_num_markers()]
            self._marker_centroids.append(np.array(
                list(map(lambda region: region.centroid, regions))))
            if self._point_in_time > self.get_wait_interval():
                try:
                    rot_axes:  List[np.float64] = []
                    for mix in range(self.get_num_markers()):
                        params, _ = sop.curve_fit(
                                f=self.opt_func,
                                xdata=self._angle_dist[:self._point_in_time],
                                ydata=np.array(self._marker_centroids)[:, mix, 1])
                        rot_axes.append(params[0])
                except RuntimeError:
                    self.info_stream(
                        "%s could not optimize parameters based on currently available data",
                        self.__class__.__name__
                    )
                    self.info_stream("%s: Rotation axis: %f", self.__class__.__name__, np.nan)
                self.info_stream("%s: Rotation axis: %f", self.__class__.__name__, np.median(rot_axes))


if __name__ == "__main__":
    pass
