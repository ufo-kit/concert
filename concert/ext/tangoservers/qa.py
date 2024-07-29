"""
qa.py
-----
Implements a device server to execute quality assurance routines during acquisition.
"""
from typing import List, AsyncIterator
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

    sigma = attribute(
        label="Sigma",
        dtype=float,
        access=AttrWriteType.READ_WRITE,
        fget="get_sigma",
        fset="set_sigma",
        doc="Standard deviation(sigma) for Gaussian low pass filtering"
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

    wait_window = attribute(
        label="Wait_Window",
        dtype=int,
        access=AttrWriteType.READ_WRITE,
        fget="get_wait_window",
        fset="set_wait_window",
        doc="initial wait time before starting rotation axis estimation"
    )

    estm_offset = attribute(
        label="Estm_Offset",
        dtype=int,
        access=AttrWriteType.READ_WRITE,
        fget="get_estm_offset",
        fset="set_estm_offset",
        doc="offset in number of projections to be used to run estimation"
    )

    _marker_centroids: List[ArrayLike]
    _point_in_time: int
    _angle_dist: ArrayLike
    _norm_buffer: List[ArrayLike]
    _estm_rot_axis: List[float]
    _monitor_window: int  # TODO: Temporary internal attribute

    async def init_device(self) -> None:
        await super().init_device()
        self._marker_centroids = []
        self._point_in_time = 0
        self._norm_buffer = []
        self._estm_rot_axis = []
        self._monitor_window = 100  # TODO: Temporary internal attribute
        self.info_stream("%s initialized device with state: %s",
                         self.__class__.__name__, self.get_state())

    def get_sigma(self) -> int:
        return self._sigma

    def set_sigma(self, sigma: float) -> None:
        self._sigma = sigma
        print(f"Sigma: {sigma}")
        self.info_stream(
                "%s has set sigma to: %f, state: %s",
            self.__class__.__name__, self._sigma, self.get_state()
        )

    def get_num_markers(self) -> int:
        return self._num_markers

    def set_num_markers(self, num_markers: int) -> None:
        self._num_markers = num_markers
        self.info_stream(
            "%s has set number of markers to: %d, state: %s",
            self.__class__.__name__, self._num_markers, self.get_state()
        )

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

    def get_wait_window(self) -> int:
        return self._wait_window

    def set_wait_window(self, wait_window: int) -> None:
        self._wait_window = wait_window
        self.info_stream(
            "%s has set initial wait window for estimation to: %d projections, state: %s",
            self.__class__.__name__, self._wait_window, self.get_state()
        )

    def get_estm_offset(self) -> int:
        return self._estm_offset

    def set_estm_offset(self, estm_offset: int) -> None:
        self._estm_offset = estm_offset
        self.info_stream(
            "%s has set estimation offset interval to: %d projections, state %s",
            self.__class__.__name__, self._estm_offset, self.get_state()
        )

    @property
    def _norm_window(self) -> int:
        """
        Denotes total number of darks and flat fields combined. These images
        represent the ingredients for normalizing the projections. This is a
        convenient property to distinguish real projections from everything else
        in the stream of incoming images in terms of point in time from the start
        of the stream.
        """
        return self._num_darks + self._num_flats

    @property
    def _gaussian_kernel_size(self) -> int:
        """Computes Gaussian kernel size from provided standard deviation(sigma)"""
        return 2 * np.ceil(3 * self._sigma).astype(int) + 1

    @DebugIt()
    @command()
    async def prepare_angular_distribution(self) -> None:
        """Prepares projected angular distribution as input values for nonlinear polynomial fit"""
        self._angle_dist = np.linspace(0., self._rot_angle, self._num_radios)
        self.info_stream("%s: prepared angular distribution", self.__class__.__name__)

    @DebugIt()
    @command()
    async def estimate_center_of_rotation(self) -> None:
        """Estimates the center of rotation"""
        await self._process_stream(self._estimate_center(self._receiver.subscribe()))

    @staticmethod
    def opt_func(angle_x: np.float64, center_p1: np.float64, radius_p2: np.float64,
                 phase_p3: np.float64) -> np.float64:
        """Defines the model function for nonlinear polynomial fit"""
        return center_p1 + radius_p2 * np.cos(angle_x + phase_p3)

    async def _estimate_center(self, producer: AsyncIterator[ArrayLike]) -> None:
        async for image in producer:
            # Process dark and flat fields
            self._point_in_time += 1
            if self._point_in_time <= self._norm_window:
                self._norm_buffer.append(image)
                self.info_stream("%s processed %d images for dark and flat fields",
                                 self.__class__.__name__, len(self._norm_buffer))
            else:
                break
        dark: ArrayLike = np.array(self._norm_buffer[:self._num_darks]).mean(axis=0)
        flat: ArrayLike = np.array(self._norm_buffer[self._num_darks:self._num_flats]).mean(axis=0)
        # Instantiate Ufo processes for flat-field correction and Gaussian filtering
        ffc = FlatCorrect(dark=dark, flat=flat, absorptivity=False)
        gb = GaussianBlur(kernel_size=self._gaussian_kernel_size, sigma=self._sigma)
        async for projection in gb(ffc(producer)):
            # Process projections
            thresh: ArrayLike = skf.threshold_multiotsu(projection, classes=3)
            mask: ArrayLike = projection < thresh.min()
            labels: ArrayLike = sms.label(mask)
            regions: List[RegionProperties] = sms.regionprops(label_image=labels)
            regions = sorted(regions, key=lambda region: region.area, reverse=True)
            # TODO: Here we implicitly assume that the detected number of regions would always be
            # greater than the expected number of markers. In practice this is an edge case because
            # it might not always be true.
            if len(regions) > self._num_markers:
                regions = regions[:self._num_markers]
            self._marker_centroids.append(np.array(
                list(map(lambda region: region.centroid, regions))))
            # We start estimating after initial wait window is passed. This is required for the
            # optimization method to be able to yield some values for the parameters.
            if self._point_in_time - self._norm_window < self._wait_window:
                self.info_stream(
                    "%s: Skipping estimation until [%d/%d]",
                    self.__class__.__name__,
                    self._point_in_time - self._norm_window,
                    self._wait_window
                )
            else:
                # We use an offset in terms of number of projections for estimation to gain speed
                # up in the process.
                if (self._point_in_time - self._norm_window) % self._estm_offset == 0:
                    try:
                        rot_axes:  List[np.float64] = []
                        # TODO: This is a potential edge case. Can this be guaranteed that the
                        # segmentation algorithm would always be able to detect desired number of
                        # markers and give us their centroids ?
                        # OR, a better idea could be that we infer the detected number of markers
                        # from the shape of the accumulation array. That actually could introduce
                        # errors in estimation. This needs to be explored and experimented. Ideal
                        # solution is the simplest one that works reasonably well all the time and
                        # not perfectly only some times.
                        for mix in range(self._num_markers):
                            x: ArrayLike = self._angle_dist[:self._point_in_time - self._norm_window]
                            y: ArrayLike = np.array(self._marker_centroids)[:, mix, 1]
                            params, _ = sop.curve_fit(f=self.opt_func,
                                                      xdata=np.squeeze(x),
                                                      ydata=np.squeeze(y))
                            rot_axes.append(params[0])
                        self.info_stream(
                            "%s: Estimated rotation axis with projections [%d/%d]: %f",
                            self.__class__.__name__,
                            self._point_in_time - self._norm_window,
                            self._num_radios,
                            np.median(rot_axes)
                        )
                        self._estm_rot_axis.append(np.median(rot_axes))
                        if (self._point_in_time - self._norm_window) % self._monitor_window == 0:
                            self.info_stream(
                                "Estimated gradient: %s",
                                np.round(np.abs(np.gradient(self._estm_rot_axis)), decimals=4)
                            )
                    except RuntimeError:
                        self.info_stream(
                            "%s could not find optimal parameters with projection: [%d/%d]",
                            self.__class__.__name__,
                            self._point_in_time - self._norm_window,
                            self._num_radios
                        )
            self._point_in_time += 1


if __name__ == "__main__":
    pass
