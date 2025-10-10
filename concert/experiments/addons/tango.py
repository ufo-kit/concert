import asyncio
import logging
import os
import numpy as np
import time

from concert.config import DISTRIBUTED_TANGO_TIMEOUT
from concert.coroutines.base import background
from concert.experiments.addons import base
from concert.experiments.base import Consumer
from concert.quantities import q
from typing import Awaitable


LOG = logging.getLogger(__name__)


class RemoteConsumer(Consumer):
    """
    A wrapper for turning coroutine functions into coroutines.

    :param corofunc: a consumer coroutine function
    :param corofunc_args: a list or tuple of *corofunc* arguemnts
    :param corofunc_kwargs: a list or tuple of *corofunc* keyword arguemnts
    """
    def __init__(self, endpoint, corofunc, corofunc_args=(), corofunc_kwargs=None):
        super().__init__(corofunc, corofunc_args=corofunc_args, corofunc_kwargs=corofunc_kwargs)
        self.endpoint = endpoint

    @property
    def remote(self):
        return True

    async def __call__(self):
        st = time.perf_counter()
        try:
            await self.corofunc(*self.args, **self.kwargs)
        except BaseException as e:
            LOG.debug(
                "`%s' occured in %s, remote cancelled with result: %s",
                e.__class__.__name__,
                self.corofunc.__qualname__,
                await asyncio.gather(self.cancel_remote(), return_exceptions=True)
            )
            raise
        finally:
            LOG.debug('%s finished in %.3f s', self.corofunc.__qualname__, time.perf_counter() - st)

    def connect_endpoint(self):
        pass

    def disconnect_endpoint(self):
        pass

    def cancel_remote(self):
        """If local task gets cancelled, this functions makes sure the error propagates to the
        remote node.
        """
        pass


class TangoConsumer(RemoteConsumer):
    """
    A wrapper for turning coroutine functions into coroutines.

    :param corofunc: a consumer coroutine function
    :param corofunc_args: a list or tuple of *corofunc* arguemnts
    :param corofunc_kwargs: a list or tuple of *corofunc* keyword arguemnts
    """
    def __init__(self, tango_device, endpoint, corofunc, corofunc_args=(), corofunc_kwargs=None):
        super().__init__(
            endpoint, corofunc, corofunc_args=corofunc_args, corofunc_kwargs=corofunc_kwargs
        )
        self._device = tango_device

    async def connect_endpoint(self):
        if (await self._device.read_attribute("endpoint")).value != self.endpoint.client_endpoint:
            await self.disconnect_endpoint()
            await self._device.write_attribute("endpoint", self.endpoint.client_endpoint)
        await self._device.connect_endpoint()

    async def disconnect_endpoint(self):
        await self._device.disconnect_endpoint()

    async def cancel_remote(self):
        await self._device.cancel()


class LiveViewConsumer(RemoteConsumer):
    """
    A wrapper for turning coroutine functions into coroutines.

    :param corofunc: a consumer coroutine function
    :param corofunc_args: a list or tuple of *corofunc* arguemnts
    :param corofunc_kwargs: a list or tuple of *corofunc* keyword arguemnts
    """
    def __init__(self, viewer, endpoint, corofunc, corofunc_args=(), corofunc_kwargs=None):
        super().__init__(
            endpoint, corofunc, corofunc_args=corofunc_args, corofunc_kwargs=corofunc_kwargs
        )
        self._viewer = viewer

    async def connect_endpoint(self):
        if await self._viewer.get_limits() == 'stream':
            self._viewer.unsubscribe()
            # Force viewer to update the limits by unsubscribing and re-subscribing after
            # setting limits to stream
            await self._viewer.set_limits('stream')
        self._viewer.subscribe(self.endpoint.client_endpoint)

    async def disconnect_endpoint(self):
        self._viewer.unsubscribe()


class TangoMixin:

    """TangoMixin does not need a producer becuase the backend processes image streams which do not
    come via concert.
    """
    async def __ainit__(self, device, endpoints: dict) -> Awaitable:
        self._device = device
        self._device.set_timeout_millis(DISTRIBUTED_TANGO_TIMEOUT)
        self._endpoints = endpoints

    async def _teardown(self):
        await self._device.disconnect_endpoint()


class Benchmarker(TangoMixin, base.Benchmarker):

    async def __ainit__(self, experiment, device, endpoints, acquisitions=None):
        await TangoMixin.__ainit__(self, device, endpoints)
        await base.Benchmarker.__ainit__(self, experiment=experiment, acquisitions=acquisitions)

    def _make_consumers(self, acquisitions):
        consumers = {}

        for acq in acquisitions:
            consumers[acq] = TangoConsumer(
                self._device, self._endpoints[acq], self.start_timer, corofunc_args=(acq.name,)
            )

        return consumers

    async def start_timer(self, acquisition_name):
        await self._device.start_timer(acquisition_name)

    async def _get_duration(self, acquisition_name):
        return (await self._device.get_duration(acquisition_name)) * q.s

    async def _teardown(self):
        await super()._teardown()
        await self._device.reset()


class ImageWriter(TangoMixin, base.ImageWriter):

    async def __ainit__(self, experiment, endpoints, acquisitions=None):
        await TangoMixin.__ainit__(self, experiment.walker.device, endpoints)
        await base.ImageWriter.__ainit__(self, experiment=experiment, acquisitions=acquisitions)

    def _make_consumers(self, acquisitions):
        """Attach all acquisitions."""
        consumers = {}

        def prepare_wrapper(name):
            async def write_sequence():
                # Make sure the directory exists
                async with self.walker:
                    # Even though acquisition name is fixed, we don't know where in the file
                    # system we are, so this must be determined dynamically when the writing
                    # is about to start
                    # This makes sure the directory exists
                    await self.walker.descend(name)
                    # Ascent immediately because walker descends to *name*
                    await self.walker.ascend()
                    # This returns a background task
                    task = self.walker.write_sequence(name)

                await task

            return write_sequence

        for acq in acquisitions:
            consumers[acq] = TangoConsumer(
                self._device, self._endpoints[acq], prepare_wrapper(acq.name)
            )

        return consumers


class LiveView(base.LiveView):

    async def __ainit__(self, viewer, endpoints, experiment, acquisitions=None):
        self.endpoints = endpoints
        await base.LiveView.__ainit__(self,
                                      viewer,
                                      experiment=experiment,
                                      acquisitions=acquisitions)

    def _make_consumers(self, acquisitions):
        consumers = {}

        async def consume():
            pass

        for acq in acquisitions:
            consumers[acq] = LiveViewConsumer(self._viewer, self.endpoints[acq], consume)

        return consumers


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


class OnlineReconstruction(TangoMixin, base.OnlineReconstruction):
    async def __ainit__(
        self,
        device,
        experiment,
        endpoints,
        acquisitions=None,
        do_normalization=True,
        average_normalization=True,
        slice_directory='online-slices',
        viewer=None
    ):
        await TangoMixin.__ainit__(self, device, endpoints)

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

    def _make_consumers(self, acquisitions):
        consumers = {}
        darks = base.get_acq_by_name(acquisitions, 'darks')
        flats = base.get_acq_by_name(acquisitions, 'flats')
        radios = base.get_acq_by_name(acquisitions, 'radios')

        if self._do_normalization:
            consumers[darks] = TangoConsumer(
                self._device, self._endpoints[darks], self.update_darks
            )
            consumers[flats] = TangoConsumer(
                self._device, self._endpoints[flats], self.update_flats
            )

        consumers[radios] = TangoConsumer(
            self._device, self._endpoints[radios], self.reconstruct
        )

        return consumers

    async def update_darks(self):
        await self._device.update_darks()

    async def update_flats(self):
        await self._device.update_flats()

    async def get_volume_shape(self):
        return await self._device.get_volume_shape()

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
        try:
            if cached:
                await self._device.rereconstruct(path)
            else:
                await self._device.reconstruct(path)
        except BaseException as e:
            self._device.cancel()
            raise

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
