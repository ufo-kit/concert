import asyncio
import functools
import inspect
import logging
from typing import Iterable, Awaitable, Optional, AsyncIterable
import numpy as np
from concert.experiments.addons import base
from concert.experiments.base import Acquisition
from concert.quantities import q
from concert.storage import RemoteDirectoryWalker
from concert.typing import AbstractTangoDevice
from concert.typing import RemoteDirectoryWalkerTangoDevice, ArrayLike

LOG = logging.getLogger(__name__)


class TangoMixin:

    """TangoMixin does not need a producer becuase the backend processes 
    image streams which do not come via concert.
    """

    remote: bool = True
    _device: AbstractTangoDevice 

    @staticmethod
    def cancel_remote(func: object) -> Awaitable:
        if not inspect.iscoroutinefunction(func):
            raise base.AddonError(
                    f"`{func.__qualname__}' is not a coroutine function")

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

    async def __ainit__(self, device: AbstractTangoDevice) -> None:
        self._device = device

    async def cancel(self) -> None:
        await self._device.cancel()

    async def _setup(self) -> None:
        await self._device.reset_connection()

    async def _teardown(self) -> None:
        await self._device.teardown()


class Benchmarker(TangoMixin, base.Benchmarker):

    async def __ainit__(self, device, acquisitions=None):
        await TangoMixin.__ainit__(self, device)
        await base.Benchmarker.__ainit__(self, acquisitions=acquisitions)

    @TangoMixin.cancel_remote
    async def start_timer(self, acquisition_name):
        await self._device.start_timer(acquisition_name)

    async def _get_duration(self, acquisition_name):
        return (await self._device.get_duration(acquisition_name)) * q.s

    async def _teardown(self):
        await super()._teardown()
        await self._device.reset()


class ImageWriter(TangoMixin, base.ImageWriter):
    """
    Implements an image writer addon which makes use of Tango device server to
    write images on remote host. The implementation of walking the filepath
    is encapsulated in the RemoteDirectoryWalker class.
    """

    async def __ainit__(self, 
                        walker: RemoteDirectoryWalker, 
                        acquisitions: Iterable[Acquisition] = None) -> None:
        await TangoMixin.__ainit__(self, walker.device)
        await base.ImageWriter.__ainit__(self, walker, 
                                         acquisitions=acquisitions)

    @TangoMixin.cancel_remote
    async def write_sequence(self, 
                             path: str, 
                             producer: AsyncIterable[ArrayLike] = None) -> None:
        # TODO: Understand the reason behind this assert statement. It could
        # be because of the wrapping of the function that cancel_remote does to
        # eventually call the function with *args and **kwargs. Need to make
        # sure the semantic behind TangoMixin class as well.
        assert producer is None
        await self.walker.write_sequence(path=path)


class LiveView(base.LiveView):

    remote = True

    async def __ainit__(self, viewer, endpoint, acquisitions=None):
        await base.LiveView.__ainit__(self, viewer, acquisitions=acquisitions)
        self._endpoint = endpoint
        self._orig_limits = await viewer.get_limits()

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
    async def __ainit__(self, device, acquisitions=None, do_normalization=True,
                        average_normalization=True, walker=None, slice_directory='online-slices'):
        await TangoMixin.__ainit__(self, device)
        await base.OnlineReconstruction.__ainit__(
            self,
            acquisitions=acquisitions,
            do_normalization=do_normalization,
            average_normalization=average_normalization,
            walker=walker,
            slice_directory=slice_directory
        )
        from concert.ext.ufo import QuantifiedProxyArgs

        self._args = await QuantifiedProxyArgs(_TangoProxyArgs(self._device))

    @TangoMixin.cancel_remote
    async def update_darks(self):
        await self._device.update_darks()

    @TangoMixin.cancel_remote
    async def update_flats(self):
        await self._device.update_flats()

    @TangoMixin.cancel_remote
    async def reconstruct(self):
        await self._device.reconstruct()

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
