import asyncio
import functools
import inspect
import logging
import os
from typing import Set, Dict
import numpy as np
import tango
from concert.experiments.addons import base
from concert.experiments.base import remote, Acquisition, Experiment
from concert.quantities import q
from concert.experiments.addons.typing import AbstractRAEDevice
from concert.experiments.addons.base import AcquisitionConsumer
from concert.helpers import CommData


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

    async def __ainit__(self, device, endpoint: CommData) -> None:
        self._device = device
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
        return (await self._device[arg]).value


class OnlineReconstruction(TangoMixin, base.OnlineReconstruction):
    
    async def __ainit__(
        self,
        device,
        experiment,
        endpoint,
        acquisitions=None,
        do_normalization=True,
        average_normalization=True,
        slice_directory='online-slices',
        viewer=None
    ):
        await TangoMixin.__ainit__(self, device, endpoint)

        # Lock the device to prevent other processes from using it
        try:
            self._device.lock()
        except tango.NonDbDevice:
            pass

        self._proxy = _TangoProxyArgs(self._device)
        await base.OnlineReconstruction.__ainit__(
            self,
            experiment=experiment,
            acquisitions=acquisitions,
            do_normalization=do_normalization,
            average_normalization=average_normalization,
            slice_directory=slice_directory,
            viewer=viewer
        )

    @TangoMixin.cancel_remote
    @remote
    async def update_darks(self):
        await self._device.update_darks()

    @TangoMixin.cancel_remote
    @remote
    async def update_flats(self):
        await self._device.update_flats()

    @TangoMixin.cancel_remote
    async def _reconstruct(self, cached=False, slice_directory=None):
        path = ""
        if self.walker:
            if (
                cached is False and await self.get_slice_directory()
                or cached is True and slice_directory
            ):
                async with self.walker:
                    path = os.path.join(
                        await self.walker.get_current(),
                        await self.get_slice_directory() if slice_directory is None \
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

    async def find_axis(self, region, z=0, store=False):
        return await self._device.find_axis([region[0], region[1], region[2], z, float(store)])

    async def get_volume(self):
        volume = np.empty(await self._device.get_volume_shape(), dtype=np.float32)
        for i in range(volume.shape[0]):
            volume[i] = await self._get_slice_z(i)

        return volume

    async def _get_slice_x(self, index):
        shape = await self._device.get_volume_shape()
        return (await self._device.get_slice_x(index)).reshape(shape[0], shape[1])

    async def _get_slice_y(self, index):
        shape = await self._device.get_volume_shape()
        return (await self._device.get_slice_y(index)).reshape(shape[0], shape[2])

    async def _get_slice_z(self, index):
        shape = await self._device.get_volume_shape()
        return (await self._device.get_slice_z(index)).reshape(shape[1], shape[2])


class RotationAxisEstimator(TangoMixin, base.Addon):
    """
    Tango addon for estimating axis of rotation during acquisition. It relies on a highly
    absorbing sphere being present at an appropriate position and a known approximate radius for
    the same.

    It encapsulates Tango device server, where remote processing of the incoming projections,
    tracking of the spheres and subsequenly estimation with curve-fitting from accumulated sphere
    centroids takes place.
    """

    async def __ainit__(self, device: AbstractRAEDevice, endpoint: CommData, experiment: Experiment,
                        acquisitions: Set[Acquisition], num_darks: int, num_flats: int,
                        num_radios: int, rot_angle: float = np.pi, **kwargs)  -> None:
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
        :param crop_vert_prop: (kwarg)vertical proportion to find the sphere, negative value denotes
        towards bottom, positive value denotes towards top, default 1, means whole projection.
        :type crop_vert_prop: int
        :param crop_left_px: (kwarg)left side cropping by pixels, default 0, means no cropping.
        :type crop_left_px: int
        :param crop_right_px: (kwarg)right side cropping by pixels, default 0, means no cropping.
        :type crop_right_px: int
        :param radius: (kwarg)approximate radius of the sphere by pixels, default 65 for 5x
        magnification (155-190 q.um Tungsten Carbide).
        :type radius: int
        :param init_wait: (kwarg)initial wait time before estimation starts, default 50 projections.
        :type init_wait: int
        :param avg_beta: (kwarg)smoothing factor to smooth out the estimated value to ensure a
        faster convergence, default 0.9.
        :param avg_beta: float
        :param diff_thresh: (kwarg)threshold diff value in estimated center of rotation from
        successive projections, default 0.1.
        :type diff_thresh: float
        :param conv_window: (kwarg)number of past estimations, based on which convergence should be
        evaluated, default 50 projections.
        :type conv_window: int
        """
        await TangoMixin.__ainit__(self, device, endpoint)
        await self._device.write_attribute("rot_angle", rot_angle)
        # Meta attributes for acquisition
        await self._device.write_attribute("attr_acq", np.array([num_darks, num_flats, num_radios],
                                                                dtype=np.int_))
        # Meta attributes for tracking
        crop_vert_prop: int = kwargs.get("crop_vert_prop", 1)
        crop_left_px: int = kwargs.get("crop_left_px", 0)
        crop_right_px: int = kwargs.get("crop_right_px", 0)
        radius: int = kwargs.get("radius", 65)
        await self._device.write_attribute("attr_track", np.array([crop_vert_prop, crop_left_px,
                                                                   crop_right_px, radius],
                                                                  dtype=np.int_))
        # Meta attributes for estimation
        init_wait: int = kwargs.get("init_wait", 50)
        avg_beta: float = kwargs.get("avg_beta", 0.9)
        diff_thresh: float = kwargs.get("diff_thresh", 0.1)
        conv_window: int = kwargs.get("conv_window", 50)
        await self._device.write_attribute("attr_estm", np.array([init_wait, avg_beta, diff_thresh,
                                                                  conv_window]))
        await base.Addon.__ainit__(self, experiment, acquisitions)

    async def _get_center_of_rotation(self) -> float:
        return await(self._device["center_of_rotation"]).value

    def _make_consumers(self, acquisitions: Set[Acquisition]) -> Dict[Acquisition,
                                                                      AcquisitionConsumer]:
        """Accumulates consumers for expected acquisitions"""
        consumers: Dict[Acquisition, AcquisitionConsumer] = {}
        consumers[base.get_acq_by_name(acquisitions, "darks")] = AcquisitionConsumer(
                self.update_darks, addon=self)
        consumers[base.get_acq_by_name(acquisitions, "flats")] = AcquisitionConsumer(
                self.update_flats, addon=self)
        consumers[base.get_acq_by_name(acquisitions, "radios")] = AcquisitionConsumer(
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
