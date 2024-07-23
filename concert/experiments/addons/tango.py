import asyncio
import functools
import inspect
import logging
import os
import numpy as np
import tango

from concert.experiments.addons import base
from concert.experiments.base import remote
from concert.quantities import q


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

    async def __ainit__(self, device):
        self._device = device

    async def cancel(self):
        await self._device.cancel()

    async def _setup(self):
        await self._device.reset_connection()

    async def _teardown(self):
        await self._device.teardown()


class Benchmarker(TangoMixin, base.Benchmarker):

    async def __ainit__(self, experiment, device, acquisitions=None):
        await TangoMixin.__ainit__(self, device)
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

    async def __ainit__(self, experiment, acquisitions=None):
        await TangoMixin.__ainit__(self, experiment.walker.device)
        await base.ImageWriter.__ainit__(self, experiment=experiment, acquisitions=acquisitions)

    @TangoMixin.cancel_remote
    @remote
    async def write_sequence(self, name):
        return await self.walker.write_sequence(name=name)


class LiveView(base.LiveView):

    async def __ainit__(self, viewer, endpoint, experiment, acquisitions=None):
        await base.LiveView.__ainit__(self, viewer, experiment=experiment, acquisitions=acquisitions)
        self._endpoint = endpoint
        self._orig_limits = await viewer.get_limits()

    @remote
    async def consume(self):
        try:
            if await self._viewer.get_limits() == 'stream':
                self._viewer.unsubscribe()
                # Force viewer to update the limits by unsubscribing and re-subscribing after
                # setting limits to stream
                await self._viewer.set_limits('stream')
                self._viewer.subscribe(self._endpoint)
        finally:
            self._orig_limits = await self._viewer.get_limits()

    async def _teardown(self):
        await super()._teardown()
        self._viewer.unsubscribe()


class _TangoProxyArgs:
    def __init__(self, device):
        self._device = device

    async def set_reco_arg(self, arg, value):
        self._device.write_attribute(arg, value)

    async def get_reco_arg(self, arg):
        return (await self._device[arg]).value


class OnlineReconstruction(TangoMixin, base.OnlineReconstruction):
    async def __ainit__(self, device, experiment, acquisitions=None, do_normalization=True,
                        average_normalization=True, slice_directory='online-slices',
                        viewer=None):
        await TangoMixin.__ainit__(self, device)

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
