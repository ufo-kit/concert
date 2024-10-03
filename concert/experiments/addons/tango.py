import asyncio
from dataclasses import dataclass
import functools
import inspect
import logging
import os
from typing import Set, Dict, Awaitable
import numpy as np
import tango
from concert.experiments.addons import base
from concert.experiments.base import remote, Acquisition, Experiment
from concert.quantities import q
from concert.experiments.addons.typing import AbstractRAEDevice
from concert.experiments.addons.base import AcquisitionConsumer
from concert.base import Parameter
from concert.ext.tangoservers.rae import EstimationAlgorithm
from concert.experiments.base import remote
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
        await base.LiveView.__ainit__(self, viewer, experiment=experiment, acquisitions=acquisitions)
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
                        await self.get_slice_directory() if slice_directory is None else slice_directory
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

    async def __ainit__(self, device: AbstractRAEDevice, endpoint: CommData, experiment: Experiment,
                        acquisitions: Set[Acquisition], num_darks: int, num_flats: int,
                        num_radios: int, rot_angle: float = np.pi, **kwargs)  -> None:
        await TangoMixin.__ainit__(self, device, endpoint)
        await self._device.write_attribute("num_darks", num_darks)
        await self._device.write_attribute("num_flats", num_flats)
        await self._device.write_attribute("num_radios", num_radios)
        await self._device.write_attribute("rot_angle", rot_angle)
        est_algo = kwargs.get("estimation_algorithm", EstimationAlgorithm.MT_SEGMENTATION)
        await self._device.write_attribute("estimation_algorithm", est_algo)
        # Process meta attributes for marker tracking method
        crop_top = kwargs.get("crop_top", 0)
        crop_bottom = kwargs.get("crop_bottom", 2016)
        crop_left = kwargs.get("crop_left", 0)
        crop_right = kwargs.get("crop_right", 2016)
        num_markers = kwargs.get("num_markers", 0)
        avg_window = kwargs.get("avg_window", 15)
        wait_window = kwargs.get("wait_window", 100)
        check_window = kwargs.get("check_window", 30)
        offset = kwargs.get("offset", 5)
        grad_thresh = kwargs.get("grad_thresh", 0.1)
        await self._device.write_attribute(
                "meta_attr_mt", np.array(
                    [crop_top, crop_bottom, crop_left, crop_right, num_markers, avg_window]))
        await self._device.write_attribute(
                "meta_attr_mt_estm", np.array(
                    [wait_window, check_window, offset, grad_thresh], dtype=np.float32))
        await self._device.prepare_angular_distribution()
        # Process meta attributes for phase correlation method
        det_row_idx = kwargs.get("det_row_idx", 0)
        num_proj_corr = kwargs.get("num_proj_corr", 200)
        await self._device.write_attribute("meta_attr_phase_corr",
                                           np.array([det_row_idx, num_proj_corr]))
        await base.Addon.__ainit__(self, experiment, acquisitions)

    async def _get_center_of_rotation(self) -> float:
        return await(self._device["center_of_rotation"]).value

    def _make_consumers(self,
                        acquisitions: Set[Acquisition]) -> Dict[Acquisition, AcquisitionConsumer]:
        """Accumulates consumers for expected acquisitions"""
        consumers: [Acquisition, AcquisitionConsumer] = {}
        consumers[base.get_acq_by_name(acquisitions, "darks")] = AcquisitionConsumer(
                self.update_darks, addon=self)
        consumers[base.get_acq_by_name(acquisitions, "flats")] = AcquisitionConsumer(
                self.update_flats, addon=self)
        consumers[base.get_acq_by_name(acquisitions, "radios")] = AcquisitionConsumer(
                self.estimate_center_of_rotation, addon=self)
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
    async def estimate_center_of_rotation(self) -> None:
        await self._device.estimate_center_of_rotation()

