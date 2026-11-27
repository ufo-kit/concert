"""Stepped, multi-dimensional image acquisition."""

import asyncio
from copy import deepcopy
from contextlib import aclosing
from concert.coroutines.base import broadcast
from concert.coroutines.sinks import count
from concert.experiments.base import Experiment, Acquisition, remote
from concert.helpers import ImageWithMetadata
from concert.processes.common import scan as scan_parameters
from concert.scanning import ScanAxis as ScanAxis, ScanMixin


__all__ = ['ScanAxis', 'ScanLogic', 'LocalScan', 'RemoteScan', 'Scan']


class _ScanAcquisition(Acquisition):
    async def _connect_local(self):
        # Wait for producer cleanup even if a consumer fails or we are cancelled.
        async with aclosing(self.producer_corofunc()) as producer:
            tasks = [asyncio.ensure_future(coro)
                     for coro in broadcast(producer, *self._consumers, count)]
            joined = asyncio.gather(*tasks)
            try:
                await asyncio.shield(joined)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                # Retrieve the original gather's exception as well.
                if joined.done() and not joined.cancelled():
                    joined.exception()


class ScanLogic(ScanMixin, Experiment):
    """Acquire one image per Cartesian combination of parameter values.

    Prefer *axes*, a dictionary mapping each name to ``(parameter, values)``.
    Dictionary insertion order defines the grid's axis order and keeps bindings
    and positions together. Alternatively, *parameters* and *values* follow
    :func:`concert.processes.common.scan`:
    pass a single bound parameter and its positions, or parallel sequences of
    parameters and arbitrary position sequences. With *order='raster'* the last
    axis changes fastest; *'snake'* reflects alternating rows in any dimension,
    and *'spiral'* visits a two-axis grid clockwise from the outside in. *names*
    optionally supplies unique axis names (otherwise ``axis_0``, ``axis_1``, ...).
    The camera is software-triggered after all parameter setters finish.

    Inspect ``await experiment.get_scan_plan()`` or ``await experiment.info_table``.
    Adjust an axis with ``await experiment.axes['x'].set_range(start, stop, step)``
    or ``await experiment.axes['x'].set_values(values)``. Use :class:`LocalScan`
    to yield images to local consumers or :class:`RemoteScan` to send images
    directly from the camera to remote consumers.
    *go_back* restores parameters on completion, failure or cancellation.

    Example::

        experiment = await LocalScan(
            axes={'x': (motor['position'], [0, 1, 2] * q.mm),
                  'exposure': (camera['exposure_time'], [1, 2] * q.ms)},
            camera=camera, go_back=True)
        await experiment.axes['x'].set_range(0 * q.mm, 5 * q.mm, 1 * q.mm)
        print(await experiment.get_scan_plan())
        await experiment.run()
    """

    async def __ainit__(self, parameters=None, values=None, *, camera, axes=None, names=None, go_back=False,
                        order='raster', walker=None, separate_runs=True, name_fmt='run_{:>04}', **kwargs):
        self._camera = camera
        scan = await _ScanAcquisition(name='scan', producer_corofunc=self.take_scan,
                                      producer=camera)
        await super().__ainit__(parameters=parameters, values=values, axes=axes, names=names, go_back=go_back,
                                order=order, acquisitions=[scan], walker=walker,
                                separate_runs=separate_runs, name_fmt=name_fmt, **kwargs)
        parameters, _ = await self._get_scan_arguments()
        if any(parameter is camera['trigger_source'] for parameter in parameters):
            raise ValueError("The scan controls the camera trigger source")

    async def _produce_scan(self, acquire_image):
        """Set positions and call the mode-specific image acquisition callback."""
        parameters, values = await self._get_scan_arguments()

        if await self._camera.get_state() == 'recording':
            await self._camera.stop_recording()
        await self._camera['trigger_source'].stash()
        try:
            await self._camera.set_trigger_source(self._camera.trigger_sources.SOFTWARE)
            async with self._camera.recording():
                async with aclosing(scan_parameters(parameters, values, acquire_image,
                                                    go_back=await self.get_go_back(),
                                                    order=await self.get_order())) as positions:
                    async for _, image in positions:
                        yield image
        finally:
            await self._camera['trigger_source'].restore()


class LocalScan(ScanLogic):
    """Acquire one image per scan position and yield it to local addons.

    Each image's ``metadata['scan']`` contains the zero-based acquisition
    ``iteration``, grid ``indices``, traversal ``order``, and ``axes`` mapping.
    Each named axis records its device ``parameter`` name and current read-back
    ``value`` (a string including units), sampled immediately before triggering.
    Existing camera metadata is preserved and metadata is JSON-serializable.
    """

    async def take_scan(self):
        """Yield images locally, waiting for cleanup when the generator closes."""
        iteration = 0

        async def acquire_image():
            nonlocal iteration
            axes = {}
            for name, axis in self.axes.items():
                parameter = await axis.get_parameter()
                axes[name] = {'parameter': parameter.name, 'value': str(await parameter.get())}
            scan_metadata = {
                'iteration': iteration,
                'indices': list(await self.get_position_indices(iteration)),
                'order': await self.get_order(),
                'axes': axes,
            }
            await self._camera.trigger()
            image = await self._camera.grab()
            metadata = deepcopy(getattr(image, 'metadata', {}))
            metadata['scan'] = scan_metadata
            iteration += 1
            return ImageWithMetadata(image, metadata=metadata)

        async with aclosing(self._produce_scan(acquire_image)) as images:
            async for image in images:
                yield image

    # Mark the generator directly: @local re-yields from an inner generator but
    # does not forward aclose(), which would defer cleanup after consumer failure.
    take_scan.remote = False


class RemoteScan(ScanLogic):
    """Acquire one image per scan position and send it to remote addons.

    The camera must support ``grab_send`` and the usual remote acquisition
    endpoint registration. Images are not transferred to the Concert session.
    """

    @remote
    async def take_scan(self):
        """Send individual images, ending the stream only at the final position."""
        total = await self.get_num_positions()
        acquired = 0

        async def acquire_image():
            nonlocal acquired
            await self._camera.trigger()
            await self._camera.grab_send(1, end=acquired == total - 1)
            acquired += 1

        async with aclosing(self._produce_scan(acquire_image)) as images:
            async for _ in images:
                pass
        return acquired


class Scan(LocalScan):
    """Compatibility name for :class:`LocalScan`."""
