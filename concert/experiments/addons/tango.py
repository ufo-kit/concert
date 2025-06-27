import asyncio
from dataclasses import dataclass
import functools
import inspect
import logging
import os
from typing import Set, Dict, Optional
import numpy as np

from concert.config import DISTRIBUTED_TANGO_TIMEOUT
from concert.coroutines.base import background
from concert.experiments.addons import base
from concert.experiments.base import remote, Acquisition, Experiment
from concert.helpers import CommData
from concert.quantities import q
from concert.typing import AbstractRAEDevice


LOG = logging.getLogger(__name__)


class TangoMixin:

    """TangoMixin does not need a producer becuase the backend processes image streams which do not
    come via concert.
    """

    @staticmethod
    def cancel_remote(func):
        if not inspect.iscoroutinefunction(func):
            raise base.AddonError(f"`{func.__qualname__}' is not a coroutine function")

        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            try:
                await func(self, *args, **kwargs)
            except BaseException as e:
                LOG.debug(
                    "`%s' occured in %s, remote cancelled with result: %s",
                    e.__class__.__name__,
                    func.__qualname__,
                    await asyncio.gather(self.cancel(), return_exceptions=True)
                )
                raise

        return wrapper

    async def __ainit__(self, device, endpoint: CommData):
        self._device = device
        self._device.set_timeout_millis(DISTRIBUTED_TANGO_TIMEOUT)
        self.endpoint = endpoint
        await self._device.write_attribute('endpoint', self.endpoint.client_endpoint)

    async def connect_endpoint(self):
        await self._device.connect_endpoint()

    async def disconnect_endpoint(self):
        await self._device.disconnect_endpoint()

    async def cancel(self):
        await self._device.cancel()

    async def _teardown(self):
        await self._device.disconnect_endpoint()

    async def get_device_uri(self) -> str:
        """Yields the device proxy uri for the tango device"""
        host: str = self._device.get_dev_host()
        port: str = self._device.get_dev_port()
        name: str = self._device.name()
        assert (host is not None and port is not None and name is not None)
        return f"{host}:{port}/{name}#dbase=no"


class Benchmarker(TangoMixin, base.Benchmarker):

    async def __ainit__(self, experiment, device, endpoint, acquisitions=None):
        await TangoMixin.__ainit__(self, device, endpoint)
        await base.Benchmarker.__ainit__(self, experiment=experiment, acquisitions=acquisitions)

    @TangoMixin.cancel_remote
    @remote
    async def start_timer(self, acquisition_name):
        await self._device.start_timer(acquisition_name)

    async def _get_duration(self, acquisition_name):
        return (await self._device.get_duration(acquisition_name)) * q.s

    async def _teardown(self):
        await super()._teardown()
        await self._device.reset()


class ImageWriter(TangoMixin, base.ImageWriter):

    async def __ainit__(self, experiment, endpoint, acquisitions=None):
        await TangoMixin.__ainit__(self, experiment.walker.device, endpoint)
        await base.ImageWriter.__ainit__(self, experiment=experiment, acquisitions=acquisitions)

    @TangoMixin.cancel_remote
    @remote
    async def write_sequence(self, name):
        return await self.walker.write_sequence(name=name)


class LiveView(base.LiveView):

    async def __ainit__(self, viewer, endpoint, experiment, acquisitions=None):
        self.endpoint = endpoint
        await base.LiveView.__ainit__(self,
                                      viewer,
                                      experiment=experiment,
                                      acquisitions=acquisitions)
        self._orig_limits = await viewer.get_limits()

    async def connect_endpoint(self):
        self._viewer.subscribe(self.endpoint.client_endpoint)

    async def disconnect_endpoint(self):
        self._viewer.unsubscribe()

    @remote
    async def consume(self):
        try:
            if await self._viewer.get_limits() == 'stream':
                self._viewer.unsubscribe()
                # Force viewer to update the limits by unsubscribing and re-subscribing after
                # setting limits to stream
                await self._viewer.set_limits('stream')
                self._viewer.subscribe(self.endpoint.client_endpoint)
        finally:
            self._orig_limits = await self._viewer.get_limits()


class _TangoProxyArgs:
    def __init__(self, device):
        self._device = device

    async def set_reco_arg(self, arg, value):
        self._device.write_attribute(arg, value)

    async def get_reco_arg(self, arg):
        if arg in ["z_parameters", "slice_metrics"]:
            func = getattr(self._device, "get_" + arg)
            return await func()
        return (await self._device[arg]).value

    async def get_parameters(self):
        flattened = await self._device.get_parameters()
        result = []
        for i in range(len(flattened) // 2):
            result.append((flattened[2 * i], flattened[2 * i + 1]))

        return tuple(result)


@dataclass(init=True, repr=True)
class RotationAxisEstimatorConfigs:
    """
    Configuration class for the RotationAxisEstimator addon. It holds default values for various
    parameters used in the addon.

    :attribute crop_vert_prop: vertical proportion to find the sphere, negative value denotes 
    towards bottom, positive value denotes towards top. It prevents searching for the sphere in
    the complete projection. It is especially relevant in context of, whether the projection
    coming out of the camera is flipped or not. Defaults to +1, means whole projection.
    :attribute crop_left_px: left side cropping by pixels, default 0, means no cropping.
    :attribute crop_right_px: right side cropping by pixels, default 0, means no cropping.
    :attribute radius: approximate radius of the sphere by pixels, default 65 for 5x
    magnification (155-190 q.um Tungsten Carbide).
    :attribute init_wait: initial wait time before estimation starts, default 50 projections.
    :attribute conv_window: number of past estimations, based on which convergence should be
    evaluated, default 50 projections.
    """
    crop_vert_prop: int = 1
    crop_left_px: int = 0
    crop_right_px: int = 0
    radius: int = 65  # Default radius in pixels, corresponds to 5x
    init_wait: int = 50  # Default initial wait time in projections
    conv_window: int = 50  # Default convergence window in projections


class RotationAxisEstimator(TangoMixin, base.Addon):
    """
    Tango addon for estimating axis of rotation during acquisition. It relies on a highly
    absorbing sphere being present at an appropriate position and a known approximate radius for
    the same. It encapsulates Tango device server, where remote processing of the incoming
    projections, tracking of the spheres and subsequenly estimation with curve-fitting from
    accumulated sphere centroids takes place.
    """

    async def __ainit__(self, device: AbstractRAEDevice, endpoint: CommData, experiment: Experiment,
                        acquisitions: Set[Acquisition], num_darks: int, num_flats: int,
                        num_radios: int, rot_angle: float = np.pi,
                        configs: RotationAxisEstimatorConfigs = RotationAxisEstimatorConfigs()
                        ) -> None:
        """
        :param device: tango device server proxy for rotation axis estimation.
        :type device: `concert.experiments.addons.typing.AbstractRAEDevice`
        :param endpoint: remote communication endpoint to receive stream.
        :type endpoint: `concert.helpers.CommData`
        :param experiment: concert experiment subclass.
        :type experiment: `concert.experiments.base.Experiment`
        :param acquisitions: acquisitions encapsulated by esperiment.
        :type acquisitions: Set[`concert.experiments.base.Acquisition`]
        :param num_darks: number of dark field projections.
        :type num_darks: int
        :param num_flats: number of flat field projections.
        :type num_flats: int
        :param num_radios: number of radiogram projections.
        :type num_radios: int
        :param rot_angle: overall rotation angle.
        :type rot_angle: float, defaults to `np.pi`
        :param configs: configuration object for the rotation axis estimator.
        :type configs: `RotationAxisEstimatorConfigs`
        """
        await TangoMixin.__ainit__(self, device, endpoint)
        await self._device.write_attribute("rot_angle", rot_angle)
        # Meta attributes for acquisition
        await self._device.write_attribute("attr_acq", np.array([num_darks, num_flats, num_radios],
                                                                dtype=np.int_))
        # Meta attributes for tracking
        await self._device.write_attribute("attr_track", np.array([
            configs.crop_vert_prop, configs.crop_left_px, configs.crop_right_px,configs.radius]))
        # Meta attributes for estimation
        await self._device.write_attribute("attr_estm", np.array([
            configs.init_wait, configs.conv_window]))
        await base.Addon.__ainit__(self, experiment, acquisitions)

    async def _get_center_of_rotation(self) -> float:
        return await(self._device["center_of_rotation"]).value

    def _make_consumers(self, acquisitions: Set[Acquisition]) -> Dict[Acquisition,
                                                                      base.AcquisitionConsumer]:
        """Accumulates consumers for expected acquisitions"""
        consumers: Dict[Acquisition, base.AcquisitionConsumer] = {}
        consumers[base.get_acq_by_name(acquisitions, "darks")] = base.AcquisitionConsumer(
                self.update_darks, addon=self)
        consumers[base.get_acq_by_name(acquisitions, "flats")] = base.AcquisitionConsumer(
                self.update_flats, addon=self)
        consumers[base.get_acq_by_name(acquisitions, "radios")] = base.AcquisitionConsumer(
                self.estimate_axis_of_rotation, addon=self)
        return consumers

    @TangoMixin.cancel_remote
    @remote
    async def update_darks(self) -> None:
        await self._device.update_darks()

    @TangoMixin.cancel_remote
    @remote
    async def update_flats(self) -> None:
        await self._device.update_flats()

    @TangoMixin.cancel_remote
    @remote
    async def estimate_axis_of_rotation(self) -> None:
        await self._device.estimate_axis_of_rotation()


class OnlineReconstruction(TangoMixin, base.OnlineReconstruction):
    """Implements tango addon for online reconstruction"""

    async def __ainit__(self, device, experiment, endpoint, acquisitions=None,
                        do_normalization=True, average_normalization=True,
                        slice_directory='online-slices', viewer=None,
                        use_rae: Optional[RotationAxisEstimator] = None):
        """
        :param device: tango device server proxy for rotation axis estimation.
        :type device: `concert.experiments.addons.typing.AbstractRAEDevice`
        :param endpoint: remote communication endpoint to receive stream.
        :type endpoint: `concert.helpers.CommData`
        :param experiment: concert experiment subclass.
        :type experiment: `concert.experiments.base.Experiment`
        :param acquisitions: acquisitions encapsulated by esperiment.
        :type acquisitions: Set[`concert.experiments.base.Acquisition`]
        :param do_normalization: whether projections would be normalized.
        :type do_normalization: bool
        :param average_normalization: whether average of the flat-fields should be used to
        normalize.
        :type average_normalization: bool
        :param slice_directory: directory to write online reconstruction.
        :type slice_directory: str
        :param viewer: viewer object to display slices.
        :type viewer: `concert.ext.viewers.ViewrBase`
        :param use_rae: optional rotation axis estimator to use for reconstruciton, when not
        provided online reconstruction tango server works standalone.
        :type use_rae: RotationAxisEstimator
        """
        await TangoMixin.__ainit__(self, device, endpoint)
        await base.OnlineReconstruction.__ainit__(
            self,
            _TangoProxyArgs(self._device),
            experiment=experiment,
            acquisitions=acquisitions,
            do_normalization=do_normalization,
            average_normalization=average_normalization,
            slice_directory=slice_directory,
            viewer=viewer
        )
        if use_rae is not None:
            rae_dev_uri: str = await use_rae.get_device_uri()
            await self._device.register_axis_feedback(rae_dev_uri)
            LOG.info("%s: registered to use rotation axis estimation", self.__class__.__name__)
        else:
            LOG.info("%s: operates on standalone mode", self.__class__.__name__)

    @TangoMixin.cancel_remote
    @remote
    async def update_darks(self):
        await self._device.update_darks()

    @TangoMixin.cancel_remote
    @remote
    async def update_flats(self):
        await self._device.update_flats()

    async def get_volume_shape(self):
        return await self._device.get_volume_shape()

    @TangoMixin.cancel_remote
    async def _reconstruct(self, cached=False, slice_directory=None):
        path = ""
        if self.walker and not await self.get_slice_metric():
            if (
                cached is False and await self.get_slice_directory()
                or cached is True and slice_directory
            ):
                async with self.walker:
                    path = os.path.join(
                        await self.walker.get_current(),
                        await self.get_slice_directory() if slice_directory is None
                        else slice_directory
                    )
        if cached:
            await self._device.rereconstruct(path)
        else:
            await self._device.reconstruct(path)

    @remote
    async def reconstruct(self):
        await base.OnlineReconstruction.reconstruct(self)

    async def _rereconstruct(self, slice_directory=None):
        await self._reconstruct(cached=True, slice_directory=slice_directory)

    @background
    async def find_parameter(self, parameter, region, metric='sag', z=None, store=False):
        region = region.to(self.UNITS[parameter.replace('-', '_')]).magnitude.tolist()
        blob = (
            "find_parameter_args",
            {
                "parameter": parameter,
                "region": region,
                "z": 0 if z is None else z.magnitude,
                "store": store,
                "metric": metric,
            }
        )
        await self._device.write_pipe("find_parameter_args", blob)
        return await self._device.find_parameter()

    async def get_volume(self):
        if len(await self.get_volume_shape()) == 1:
            volume = await self._device.get_volume_line()
        else:
            volume = np.empty(await self.get_volume_shape(), dtype=np.float32)
            for i in range(volume.shape[0]):
                volume[i] = await self._get_slice_z(i)

        return volume

    async def _get_slice_x(self, index):
        shape = await self.get_volume_shape()
        return (await self._device.get_slice_x(index)).reshape(shape[0], shape[1])

    async def _get_slice_y(self, index):
        shape = await self.get_volume_shape()
        return (await self._device.get_slice_y(index)).reshape(shape[0], shape[2])

    async def _get_slice_z(self, index):
        shape = await self.get_volume_shape()
        return (await self._device.get_slice_z(index)).reshape(shape[1], shape[2])

