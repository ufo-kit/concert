"""
qa.py
-----
Implements a device server to execute quality assurance routines during
acquisition.
"""
from typing import List, AsyncIterator
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
from concert.ext.ufo import FlatCorrect, InjectProcess, get_task


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
    _norm_buffer: List[ArrayLike]

    async def init_device(self) -> None:
        await super().init_device()
        self._marker_centroids = []
        self._point_in_time = 0
        self._norm_buffer = []
        self.info_stream("%s initialized device with state: %s",
                         self.__class__.__name__, self.get_state())

    def get_lpf_size(self) -> int:
        return self._lpf_size

    def set_lpf_size(self, lpf_size: int) -> None:
        self._lpf_size = lpf_size
        self.info_stream(
            "%s has set low pass filter size to: %d, state: %s",
            self.__class__.__name__, self.get_lpf_size(), self.get_state()
        )

    def get_num_markers(self) -> int:
        return self._num_markers

    def set_num_markers(self, num_markers: int) -> None:
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

    def get_num_darks(self) -> int:
        return self._num_darks

    def set_num_darks(self, num_darks: int) -> int:
        self._num_darks = num_darks
        self.info_stream(
            "%s has set overall number of dark field acquisitions to: %d, state: %s",
            self.__class__.__name__, self.get_num_darks(), self.get_state()
        )

    def get_num_flats(self) -> int:
        return self._num_flats

    def set_num_flats(self, num_flats: int) -> None:
        self._num_flats = num_flats
        self.info_stream(
            "%s has set overall number of flat field acquisitions to: %d, state: %s",
            self.__class__.__name__, self.get_num_flats(), self.get_state()
        )

    def get_num_radios(self) -> int:
        return self._num_radios

    def set_num_radios(self, num_radios: int) -> None:
        self._num_radios = num_radios
        self.info_stream(
            "%s has set overall number of projections to: %d, state: %s",
            self.__class__.__name__, self.get_num_radios(), self.get_state()
        )

    def get_wait_interval(self) -> int:
        return self._wait_interval

    def set_wait_interval(self, wait_interval: int) -> None:
        self._wait_interval = wait_interval
        self.info_stream(
            "%s has set wait onterval for estimation to: %d projections, state: %s",
            self.__class__.__name__, self.get_wait_interval(), self.get_state()
        )

    @property
    def _norm_interval(self) -> int:
        """
        Denotes total number of darks and flat fields combined. These images
        represent the ingredients for normalizing the projections. This is a
        convenient property to distinguish real projections from everything else
        in the stream of incoming images in terms of point in time from the start
        of the stream.
        """
        return self._num_darks + self._num_flats

    @DebugIt()
    @command()
    async def prepare_angular_distribution(self) -> None:
        self._angle_dist = np.linspace(0., self.get_rot_angle(), self.get_num_radios())
        self.info_stream("%s: prepared angular distribution", self.__class__.__name__)

    @DebugIt()
    @command()
    async def derive_rot_axis(self) -> None:
        # TODO: Consider estimating axis with an offset in projections. Estimating
        # axis with every projection is slow without further optimization.
        await self._process_stream(self._compute_axis_ufo(self._receiver.subscribe()))

    @staticmethod
    def opt_func(angle_x: np.float64,
                 center_p1: np.float64,
                 radius_p2: np.float64,
                 phase_p3: np.float64) -> np.float64:
        return center_p1 + radius_p2 * np.cos(angle_x + phase_p3)

    async def _compute_axis_ufo(self, producer: AsyncIterator[ArrayLike]) -> None:
        async for image in producer:
            # Process normalization images
            self._point_in_time += 1
            if self._point_in_time <= self._norm_interval:
                self._norm_buffer.append(image)
                self.info_stream("%s processed %d images for dark and flat fields",
                                 self.__class__.__name__, len(self._norm_buffer))
            else:
                break
        dark: ArrayLike = np.array(self._norm_buffer[:self._num_darks]).mean(axis=0)
        flat: ArrayLike = np.array(self._norm_buffer[self._num_darks:self._num_flats]).mean(axis=0)
        ffc = FlatCorrect(dark=dark, flat=flat, absorptivity=False)
        async for projection in ffc(producer):
            # Process projections
            fc = snd.median_filter(projection, size=self.get_lpf_size())
            thresh: ArrayLike = skf.threshold_multiotsu(fc, classes=3)
            mask: ArrayLike = fc < thresh.min()
            labels: ArrayLike = sms.label(mask)
            regions: List[RegionProperties] = sms.regionprops(label_image=labels)
            regions = sorted(regions, key=lambda region: region.area, reverse=True)
            if len(regions) > self.get_num_markers():
                regions = regions[:self.get_num_markers()]
            self._marker_centroids.append(np.array(
                list(map(lambda region: region.centroid, regions))))
            try:
                if self._point_in_time - self._norm_interval > self.get_wait_interval():
                    rot_axes:  List[np.float64] = []
                    # TODO: Think about the edge cases. Can this be guaranteed
                    # that the segmentation algorithm would always be able to
                    # detect desired number of markers and give us their centroids ?
                    for mix in range(self.get_num_markers()):
                        x: ArrayLike = self._angle_dist[:self._point_in_time - self._norm_interval]
                        y: ArrayLike = np.array(self._marker_centroids)[:, mix, 1]
                        params, _ = sop.curve_fit(f=self.opt_func,
                                                  xdata=np.squeeze(x),
                                                  ydata=np.squeeze(y))
                        rot_axes.append(params[0])
                    self.info_stream(
                        "%s: Estimated rotation axis with projections [%d/%d]: %f",
                        self.__class__.__name__,
                        self._point_in_time - self._norm_interval,
                        self._num_radios,
                        np.median(rot_axes)
                    )
                else:
                    self.info_stream(
                        "%s: Skipping estimation until [%d/%d]",
                        self.__class__.__name__,
                        self._point_in_time - self._norm_interval,
                        self._wait_interval
                    )
            except RuntimeError:
                self.info_stream(
                    "%s could not optimize parameters so far with projection: [%d/%d]",
                    self.__class__.__name__,
                    self._point_in_time - self._norm_interval,
                    self._num_radios
                )
            self._point_in_time += 1

    async def _compute_axis_numpy(self, producer: AsyncIterator[ArrayLike]) -> None:
        async for image in producer:
            self._point_in_time += 1
            if self._point_in_time <= self._norm_interval:
                self._norm_buffer.append(image)
                self.info_stream("%s processed %d images for dark and flat fields",
                                 self.__class__.__name__, len(self._norm_buffer))
            else:
                dark: ArrayLike = np.array(self._norm_buffer[:self._num_darks]).mean(axis=0)
                flat: ArrayLike = np.array(self._norm_buffer[self._num_darks:self._num_flats]).mean(axis=0)
                fc: ArrayLike = np.nan_to_num(
                    (image - dark)/(flat - dark),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0
                )
                fc = snd.median_filter(fc, size=self.get_lpf_size())
                thresh: ArrayLike = skf.threshold_multiotsu(fc, classes=3)
                mask: ArrayLike = fc < thresh.min()
                labels: ArrayLike = sms.label(mask)
                regions: List[RegionProperties] = sms.regionprops(label_image=labels)
                regions = sorted(regions, key=lambda region: region.area, reverse=True)
                if len(regions) > self.get_num_markers():
                    regions = regions[:self.get_num_markers()]
                self._marker_centroids.append(np.array(
                    list(map(lambda region: region.centroid, regions))))
                try:
                    if self._point_in_time - self._norm_interval > self.get_wait_interval():
                        rot_axes:  List[np.float64] = []
                        # TODO: Think about the edge cases. Can this be guaranteed
                        # that the segmentation algorithm would always be able to
                        # detect desired number of markers and give us their centroids ?
                        for mix in range(self.get_num_markers()):
                            x: ArrayLike = self._angle_dist[:self._point_in_time - self._norm_interval]
                            y: ArrayLike = np.array(self._marker_centroids)[:, mix, 1]
                            params, _ = sop.curve_fit(f=self.opt_func,
                                                      xdata=np.squeeze(x),
                                                      ydata=np.squeeze(y))
                            rot_axes.append(params[0])
                        self.info_stream(
                            "%s: Estimated rotation axis with projections [%d/%d]: %f",
                            self.__class__.__name__,
                            self._point_in_time - self._norm_interval,
                            self._num_radios,
                            np.median(rot_axes)
                        )
                    else:
                        self.info_stream(
                            "%s: Skipping estimation until [%d/%d]",
                            self.__class__.__name__,
                            self._point_in_time - self._norm_interval,
                            self._wait_interval
                        )
                except RuntimeError:
                    self.info_stream(
                        "%s could not optimize parameters so far with projection: [%d/%d]",
                        self.__class__.__name__,
                        self._point_in_time - self._norm_interval,
                        self._num_radios
                    )

if __name__ == "__main__":
    pass
