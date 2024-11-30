"""
align.py
--------
Implements a device server to execute alignment routines during acquisition.
"""
from enum import IntEnum
from typing import List, AsyncIterator, Tuple, Dict, Awaitable, Optional
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
from tango import DebugIt, DevState, CmdArgType, EventType, ArgType, AttrDataFormat
from tango.server import attribute, command, AttrWriteType, pipe, PipeWriteType
from concert.ext.tangoservers.base import TangoRemoteProcessing
from concert.ext.ufo import FlatCorrect
from concert.measures import rotation_axis


class AutoAligner(TangoRemoteProcessing):
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

    async def init_device(self) -> None:
        await super().init_device()
        self._algorithm = Algorithm.MARKER_TRACKING
        self._tracking = Tracking.SINGLE_MARKER_MSE
        self._angles = np.array([])
        self._axis_of_rotation = 0.
        self.set_state(DevState.STANDBY)
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

    def get_axis_of_rotation(self) -> ArrayLike:
        return self._axis_of_rotation

    def set_axis_of_rotation(self, new_value: ArrayLike) -> None:
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
        pass

if __name__ == "__main__":
    pass
