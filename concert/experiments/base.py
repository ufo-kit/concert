"""
An experiment can be run multiple times. The base :py:class:`.Experiment` takes
care of proper logging structure.
"""

import asyncio
import functools
import inspect
import logging
import os
import time
import json

import concert.devices.base
from concert.coroutines.base import background, broadcast, start
from concert.coroutines.sinks import count
from concert.progressbar import wrap_iterable
from concert.base import (
    check,
    Parameter,
    Selection,
    StateError,
    RunnableParameterizable
)
from concert.helpers import get_basename
from concert.loghandler import AsyncLoggingHandlerCloser
from functools import partial


LOG = logging.getLogger(__name__)

_runnable_state = ['standby', 'error', 'cancelled']


class Consumer:
    """
    A wrapper for turning coroutine functions into coroutines.

    :param corofunc: a consumer coroutine function
    :param corofunc_args: a list or tuple of *corofunc* arguemnts
    :param corofunc_kwargs: a list or tuple of *corofunc* keyword arguemnts
    :param addon: a :class:`~concert.experiments.addons.Addon` object
    """
    def __init__(self, corofunc, corofunc_args=(), corofunc_kwargs=None, addon=None):
        self._corofunc = corofunc
        self.addon = addon
        self.args = corofunc_args
        self.kwargs = {} if corofunc_kwargs is None else corofunc_kwargs

    @property
    def corofunc(self):
        return self._corofunc

    async def __call__(self, producer):
        corofunc = self._corofunc
        if producer:
            corofunc = partial(self._corofunc, producer)

        st = time.perf_counter()
        await corofunc(*self.args, **self.kwargs)
        LOG.debug('%s finished in %.3f s', self._corofunc.__qualname__, time.perf_counter() - st)


class Acquisition(RunnableParameterizable):
    """
    An acquisition acquires data, gets it and sends it to consumers. This is a base class for local
    and remote acquisitions and must not be used directly.

    .. py:attribute:: name

        name of this acquisition

    .. py:attribute:: producer_corofunc

        a callable with no arguments which returns a generator yielding data items once called.

    .. py:attribute:: producer
        data producer (usually a :class:`~concert.devices.cameras.base.Camera` class), must be
        specified for remote acquisitions.

    .. py:attribute:: acquire

        a coroutine function which acquires the data, takes no arguments, can be None.

    """

    async def __ainit__(self, name, producer_corofunc, producer=None, acquire=None):
        self.name = name
        self.producer = producer
        if producer_corofunc.remote:
            if producer is None:
                raise ValueError("producer must be specified for remote acquisitions")
            self._connect = self._connect_remote
            self.remote = True
        else:
            self.remote = False
            self._connect = self._connect_local
        self.producer_corofunc = producer_corofunc
        self._consumers = []
        self._workers = []
        # Don't bother with checking this for None later
        if acquire and not asyncio.iscoroutinefunction(acquire):
            raise TypeError('acquire must be a coroutine function')
        self.acquire = acquire
        await super().__ainit__()

    @background
    async def _run(self):
        """Run the acquisition, i.e. acquire the data and connect the producer and consumers."""
        LOG.debug(f"Running acquisition '{self.name}'")
        if self.acquire:
            await self.acquire()

        await self._connect()

    def contains(self, consumer):
        return consumer in self._consumers

    def add_consumer(self, consumer):
        """Add *consumer*, *remote* must match this acquisition mode."""
        if self.remote ^ consumer.corofunc.remote:
            raise ConsumerError("Cannot attach local consumers to remote producers and vice versa")
        self._consumers.append(consumer)
        LOG.debug('Adding %s to acquisition %s', consumer.__class__.__name__, self.name)

    def remove_consumer(self, consumer):
        """Remove *addon*'s consumer."""
        self._consumers.remove(consumer)
        LOG.debug('Removing %s from acquisition %s', consumer.__class__.__name__, self.name)

    async def _connect_local(self):
        """
        The implementation of feeding data to consumers, i.e. broadcast data from producer to all
        consumers.
        """
        consumers = self._consumers + [count]
        coros = broadcast(self.producer_corofunc(), *consumers)
        num = (await asyncio.gather(*coros, return_exceptions=False))[-1]
        LOG.debug("`%s' handled %d items", self.name, num)

    async def _connect_remote(self):
        """
        The implementation of feeding data to consumers, i.e. start remote consumers and once the
        data is acquired, notify them and wait for them to finish.
        """
        async def producer_coro():
            st = time.perf_counter()
            producer = self.producer_corofunc()

            # There are two scenarios:
            # 1. async generator inside which every image is sent explicitly (for fine-level
            # control)
            # 2. coroutine which just sends a bulk of images, which reduces the overhead of
            # constantly sending grab commands to the camera server.
            if inspect.isasyncgen(producer):
                i = 0
                async for _ in producer:
                    i += 1
            else:
                i = await producer
            LOG.debug("`%s': producer finished in %.3f s", self.name, time.perf_counter() - st)

        async def cancel_and_wait(tasks):
            for task in tasks:
                task.cancel()
            return await asyncio.gather(*tasks, return_exceptions=True)

        # Connect the producer to all consumers
        for consumer in self._consumers:
            if consumer.addon is not None:
                await self.producer.register_endpoint(consumer.addon.endpoint)
                await consumer.addon.connect_endpoint()

        tasks = [start(producer_coro())]
        self._producer_task = tasks[0]
        tasks += [start(consumer(None)) for consumer in self._consumers]
        LOG.debug(
            "`%s': starting producer `%s' and consumers %s",
            self.name,
            self.producer_corofunc.__qualname__,
            [consumer.corofunc.__qualname__ for consumer in self._consumers])

        self.tasks = tasks

        try:
            # What can happen while awaiting *task*:
            # 1. all tasks finish
            # 2. Someone is cancelled
            # 3. We are cancelled
            # 4. Someone raises exception
            while True:
                done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

                completed = done.pop()
                if completed.cancelled() or completed.exception():
                    await cancel_and_wait(tasks)
                    if completed.exception():
                        raise completed.exception()

                if not tasks:
                    break
        except BaseException as exception:
            # If something went wrong we cannot leave the running tasks haning, otherwise the
            # remotes might still be waiting for data, so cancel processing. Processing is
            # responsible for stopping the remote processing as well (e.g. call cancel_remote() on
            # a Tango addon)!
            pending_result = await cancel_and_wait(tasks)
            LOG.debug(
                "`%s': `%s' during remote processing, results: `%s'",
                self.name,
                exception.__class__.__name__,
                pending_result
            )
            raise
        finally:
            # No matter what happens disconnect the producers and consumers to have a clean state
            for consumer in self._consumers:
                if consumer.addon is not None:
                    await self.producer.unregister_endpoint(consumer.addon.endpoint)
                    await consumer.addon.disconnect_endpoint()

    @background
    @check(source=['standby', 'error', 'cancelled'], target=['standby', 'cancelled'])
    async def __call__(self):
        self._run_awaitable = self._run()
        await self._run_awaitable

    def __repr__(self):
        return "Acquisition({})".format(self.name)


class Experiment(RunnableParameterizable):

    """
    Experiment base class. An experiment can be run multiple times with the output data and log
    stored on disk. You can prepare every run by :meth:`.prepare` and finish the run by
    :meth:`.finish`. These methods do nothing by default. They can be useful e.g. if you need to
    reinitialize some experiment parts or want to attach some logging output.

    .. py:attribute:: acquisitions
        :noindex:

        A list of acquisitions this experiment is composed of

    .. py:attribute:: walker

       A :class:`concert.storage.Walker` descends to a data set specific for every run if given

    .. py:attribute:: separate_scans

        If True, *walker* does not descend to data sets based on specific runs

    .. py:attribute:: name_fmt

        Since experiment can be run multiple times each iteration will have a separate entry
        on the disk. The entry consists of a name and a number of the current iteration, so the
        parameter is a formattable string.

    .. py:attribute:: ready_to_prepare_next_sample

        asyncio.Event that can be used to tell a processes.experiment.Director that the next
        iteration can be prepared. Can be set() to allow the preparation while the experiment is
        still running.

    """

    iteration = Parameter()
    separate_scans = Parameter()
    name_fmt = Parameter()
    current_name = Parameter(help="Name of the current iteration")
    log_level = Selection(['critical', 'error', 'warning', 'info', 'debug'])
    log_devices_at_start = Parameter()
    log_devices_at_finish = Parameter()

    async def __ainit__(self, acquisitions, walker=None, separate_scans=True,
                        name_fmt='scan_{:>04}'):
        self._acquisitions = []
        for acquisition in acquisitions:
            self.add(acquisition)
        self.walker = walker
        self._separate_scans = separate_scans
        self._name_fmt = name_fmt
        self._current_name = ""
        self._iteration = 0
        self.log = LOG
        self._devices_to_log = {}
        self._devices_to_log_optional = {}
        self._log_devices_at_start = None
        self._log_devices_at_finish = None
        self.ready_to_prepare_next_sample = asyncio.Event()
        await super().__ainit__()
        await self.set_log_devices_at_start(True)
        await self.set_log_devices_at_finish(True)

        if separate_scans and walker:
            # The data is not supposed to be overwritten, so find an iteration which
            # hasn't been used yet
            while await self.walker.exists(self._name_fmt.format(self._iteration)):
                self._iteration += 1

    async def _set_log_devices_at_start(self, log):
        self._log_devices_at_start = bool(log)

    async def _get_log_devices_at_start(self):
        return self._log_devices_at_start

    async def _set_log_devices_at_finish(self, log):
        self._log_devices_at_finish = bool(log)

    async def _get_log_devices_at_finish(self):
        return self._log_devices_at_finish

    def add_device_to_log(self, name: str, device: concert.devices.base.Device, optional=False):
        """
        Add a device to log.

        :param name: Name of the device
        :param device: Device to log
        :param optional: If True, an exception when trying to log the device will not cause an
            error.
        """
        if optional:
            self._devices_to_log_optional[name] = device
        else:
            self._devices_to_log[name] = device

    async def _prepare_metadata_str(self) -> str:
        """Prepares the experiment metadata to be written to file. It is
        a dictionary which potentially encapsulates one or more dictionary
        objects.
        """
        metadata = {}
        exp_params = {}
        for param in self:
            exp_params[param.name] = str(await param.get())
        metadata["experiment"] = exp_params
        for name, device in self._devices_to_log.items():
            device_data = {}
            for param in device:
                device_data[param.name] = str(await param.get())
            metadata[name] = device_data

        for name, device in self._devices_to_log_optional.items():
            device_data = {}
            for param in device:
                try:
                    device_data[param.name] = str(await param.get())
                except Exception as e:
                    self.log.info(f"Error while logging optional device {name}")
                    self.log.info(e)
            metadata[name] = device_data
        return json.dumps(metadata, indent=4)

    async def _get_iteration(self):
        return self._iteration

    async def _set_iteration(self, iteration):
        self._iteration = iteration

    async def _get_current_name(self):
        return self._current_name

    async def _get_separate_scans(self):
        return self._separate_scans

    async def _set_separate_scans(self, separate_scans):
        self._separate_scans = separate_scans

    async def _get_name_fmt(self):
        return self._name_fmt

    async def _set_name_fmt(self, fmt):
        self._name_fmt = fmt

    async def _get_log_level(self):
        return logging.getLevelName(self.log.getEffectiveLevel()).lower()

    async def _set_log_level(self, level):
        self.log.setLevel(level.upper())

    async def prepare(self):
        """Gets executed before every experiment run."""
        pass

    async def finish(self):
        """Gets executed after every experiment run."""
        pass

    async def attach(self, addon):
        """Attach *addon* to all acquisitions."""
        await addon.attach(self.acquisitions)

    async def detach(self, addon):
        """Detach *addon* from all acquisitions."""
        await addon.detach(self.acquisitions)

    @property
    def acquisitions(self):
        """Acquisitions is a read-only attribute which has to be manipulated by explicit methods
        provided by this class.
        """
        return tuple(self._acquisitions)

    def add(self, acquisition):
        """
        Add *acquisition* to the acquisition list and make it accessible as
        an attribute::

            frames = Acquisition(...)
            experiment.add(frames)
            # This is possible
            experiment.frames
        """
        self._acquisitions.append(acquisition)
        setattr(self, acquisition.name, acquisition)

    def remove(self, acquisition):
        """Remove *acquisition* from experiment."""
        self._acquisitions.remove(acquisition)
        delattr(self, acquisition.name)

    def swap(self, first, second):
        """
        Swap acquisition *first* with *second*. If there are more occurrences
        of either of them then the ones which are found first in the acquisitions
        list are swapped.
        """
        if first not in self._acquisitions or second not in self._acquisitions:
            raise ValueError("Both acquisitions must be part of the experiment")

        first_index = self._acquisitions.index(first)
        second_index = self._acquisitions.index(second)
        self._acquisitions[first_index] = second
        self._acquisitions[second_index] = first

    def get_acquisition(self, name):
        """
        Get acquisition by its *name*. In case there are more like it, the first
        one is returned.
        """
        for acq in self._acquisitions:
            if acq.name == name:
                return acq
        raise ExperimentError("Acquisition with name `{}' not found".format(name))

    async def get_running_acquisition(self):
        """
        Get the currently running acquisition.
        """
        for acq in self._acquisitions:
            if await acq.get_state() == "running":
                return acq
        return None

    async def acquire(self):
        """
        Acquire data by running the acquisitions. This is the method which implements
        the data acquisition and should be overwritten if more functionality is required,
        unlike :meth:`~.Experiment.run`.
        """
        for acq in wrap_iterable(self._acquisitions):
            if await self.get_state() != 'running':
                break
            await acq()

    @background
    async def _run(self):
        self.ready_to_prepare_next_sample.clear()
        start_time = time.time()
        handler = None
        iteration = await self.get_iteration()
        separate_scans = await self.get_separate_scans()

        if self.walker:
            if separate_scans:
                await self.walker.descend((await self.get_name_fmt()).format(iteration))
            if os.path.exists(await self.walker.get_current()):
                handler: AsyncLoggingHandlerCloser = await self.walker.register_logger(
                    logger_name=self.__class__.__name__,
                    log_level=logging.NOTSET,
                    file_name="experiment.log"
                )
                self.log.addHandler(handler)
                if await self.get_log_devices_at_start():
                    exp_metadata: str = await self._prepare_metadata_str()
                    await self.walker.log_to_json(payload=exp_metadata,
                                                  filename="experiment_start.json")
            self._current_name = get_basename(await self.walker.get_current())
        self.log.info(await self.info_table)
        for name, device in self._devices_to_log.items():
            self.log.info(f"Device {name}:")
            self.log.info(await device.info_table)
        LOG.debug('Experiment iteration %d start', iteration)

        try:
            await self.prepare()
            await self.acquire()
        finally:
            try:
                await self.finish()
                if self.walker:
                    if await self.get_log_devices_at_finish():
                        exp_metadata: str = await self._prepare_metadata_str()
                        await self.walker.log_to_json(payload=exp_metadata,
                                                      filename="experiment_finish.json")
            except Exception as e:
                LOG.warning(f"Error `{e}' while finalizing experiment")
                raise StateError('error', msg=str(e))
            finally:
                self.ready_to_prepare_next_sample.set()
                if separate_scans and self.walker:
                    await self.walker.ascend()
                LOG.debug('Experiment iteration %d duration: %.2f s',
                          iteration, time.time() - start_time)
                if handler:
                    await handler.aclose()
                    self.log.removeHandler(handler)
                await self.set_iteration(iteration + 1)


class AcquisitionError(Exception):
    """Acquisition-related exceptions."""
    pass


class ConsumerError(Exception):
    """Consumer-related exceptions."""
    pass


class ExperimentError(Exception):
    """Experiment-related exceptions."""
    pass


def remote(corofunc):
    """Decorator which marks *corofunc* as remote."""
    @functools.wraps(corofunc)
    async def wrapped(*args, **kwargs):
        return await corofunc(*args, **kwargs)

    wrapped.remote = True

    return wrapped


def local(corofunc):
    """Decorator which marks *corofunc* as local. If *corofunc* is an async generator function, then
    it must yield values itself, it cannot just return a generator, otherwise it would not be
    recognized by inspect.isasyncgenfunction.
    """
    import inspect

    @functools.wraps(corofunc)
    async def wrapped_asyncgen(*args, **kwargs):
        if inspect.isasyncgenfunction(corofunc):
            # We need to re-yield, otherwise this would not be an asyncgenfunction
            async for item in corofunc(*args, **kwargs):
                yield item

    @functools.wraps(corofunc)
    async def wrapped(*args, **kwargs):
        return await corofunc(*args, **kwargs)

    wrapped.remote = False
    wrapped_asyncgen.remote = False

    if inspect.isasyncgenfunction(corofunc):
        return wrapped_asyncgen
    else:
        return wrapped
