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
from concert.base import check, Parameterizable, Parameter, Selection, State, StateError
from concert.helpers import get_state_from_awaitable
from functools import partial


LOG = logging.getLogger(__name__)

_runnable_state = ['standby', 'error', 'cancelled']


class Consumer:
    """
    A wrapper for turning coroutine functions into coroutines.

    :param corofunc: a consumer coroutine function
    :param corofunc_args: a list or tuple of *corofunc* arguemnts
    :param corofunc_kwargs: a list or tuple of *corofunc* keyword arguemnts
    """
    def __init__(self, corofunc, corofunc_args=(), corofunc_kwargs=None):
        self._corofunc = corofunc
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


class Acquisition(Parameterizable):
    """
    An acquisition acquires data, gets it and sends it to consumers. This is a base class for local
    and remote acquisitions and must not be used directly.

    .. py:attribute:: name

        name of this acquisition

    .. py:attribute:: producer_corofunc

        a callable with no arguments which returns a generator yielding data items once called.

    .. py:attribute:: acquire

        a coroutine function which acquires the data, takes no arguments, can be None.

    """
    state = State(default='standby')

    async def __ainit__(self, name, producer_corofunc, acquire=None):
        self.name = name
        if producer_corofunc.remote:
            self._connect = self._connect_remote
            self.remote = True
        else:
            self.remote = False
            self._connect = self._connect_local
        self.producer = producer_corofunc
        self._consumers = []
        self._workers = []
        # Don't bother with checking this for None later
        if acquire and not asyncio.iscoroutinefunction(acquire):
            raise TypeError('acquire must be a coroutine function')
        self.acquire = acquire
        self._run_awaitable = None
        await Parameterizable.__ainit__(self)

    async def _get_state(self):
        return await get_state_from_awaitable(self._run_awaitable)

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
        coros = broadcast(self.producer(), *consumers)
        num = (await asyncio.gather(*coros, return_exceptions=False))[-1]
        # await asyncio.gather(*(consumer.proxy.wait(num) for consumer in self._consumers))

    async def _connect_remote(self):
        """
        The implementation of feeding data to consumers, i.e. start remote consumers and once the
        data is acquired, notify them and wait for them to finish.
        """
        async def producer_coro():
            st = time.perf_counter()
            producer = self.producer()

            # There are two scenarios:
            # 1. async generator inside which every image is sent explicitly (for fine-level
            # control)
            # 2. coroutine which just sends a bulk of images, which reduces the overhead of
            # constantly sending grab commands to the camera server.
            if inspect.isasyncgen(producer):
                i = 0
                async for nothing in producer:
                    i += 1
            else:
                i = await producer
            LOG.debug("`%s': producer finished in %.3f s", self.name, time.perf_counter() - st)

        async def cancel_and_wait(tasks):
            for task in tasks:
                task.cancel()
            return await asyncio.gather(*tasks, return_exceptions=True)

        tasks = [start(producer_coro())]
        self._producer_task = tasks[0]
        tasks += [start(consumer(None)) for consumer in self._consumers]
        LOG.debug(
            "`%s': starting producer `%s' and consumers %s",
            self.name,
            self.producer.__qualname__,
            [consumer.corofunc.__qualname__ for consumer in self._consumers])

        try:
            # What can happen while awaiting *task*:
            # 1. all tasks finish
            # 2. Someone is cancelled
            # 3. We are cancelled
            # 4. Someone raises exception
            while True:
                done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                self._pending_tasks = tasks

                if not tasks:
                    break

                completed = done.pop()
                if completed.cancelled() or completed.exception():
                    await cancel_and_wait(tasks)
                    if completed.exception():
                        raise completed.exception()
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

    @background
    @check(source=['standby', 'error', 'cancelled'], target=['standby', 'cancelled'])
    async def __call__(self):
        self._run_awaitable = self._run()
        await self._run_awaitable

    def __repr__(self):
        return "Acquisition({})".format(self.name)


class Experiment(Parameterizable):

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
    state = State(default='standby')
    log_level = Selection(['critical', 'error', 'warning', 'info', 'debug'])

    async def __ainit__(self, acquisitions, walker=None, separate_scans=True,
                        name_fmt='scan_{:>04}'):
        self._acquisitions = []
        for acquisition in acquisitions:
            self.add(acquisition)
        self.walker = walker
        self._separate_scans = separate_scans
        self._name_fmt = name_fmt
        self._iteration = 0
        self.log = LOG
        self._devices_to_log = {}
        self.ready_to_prepare_next_sample = asyncio.Event()
        self._run_awaitable = None
        await Parameterizable.__ainit__(self)

        if separate_scans and walker:
            # The data is not supposed to be overwritten, so find an iteration which
            # hasn't been used yet
            while await self.walker.exists(self._name_fmt.format(self._iteration)):
                self._iteration += 1

    def add_device_to_log(self, name: str, device: concert.devices.base.Device):
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
        return json.dumps(metadata, indent=4)

    async def _get_iteration(self):
        return self._iteration

    async def _set_iteration(self, iteration):
        self._iteration = iteration

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
    @check(source=['standby', 'error', 'cancelled'], target=['standby', 'cancelled'])
    async def run(self):
        self._run_awaitable = self._run()
        await self._run_awaitable

    async def _get_state(self):
        return await get_state_from_awaitable(self._run_awaitable)

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
                handler = await self.walker.get_log_handler()
                self.log.addHandler(handler)
                exp_metadata: str = await self._prepare_metadata_str()
                await self.walker.log_to_json(payload=exp_metadata)
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


def remote(func):
    """Decorator which marks *func* as remote."""
    @functools.wraps(func)
    async def wrapped(*args, **kwargs):
        return await func(*args, **kwargs)

    wrapped.remote = True

    return wrapped


def local(func):
    """Decorator which marks *func* as local. If *func* is an async generator function, then it must
    yield values itself, it cannot just return a generator, otherwise it would not be recognized by
    inspect.isasyncgenfunction.
    """
    import inspect

    @functools.wraps(func)
    async def wrapped_asyncgen(*args, **kwargs):
        if inspect.isasyncgenfunction(func):
            # We need to re-yield, otherwise this would not be an asyncgenfunction
            async for item in func(*args, **kwargs):
                yield item

    @functools.wraps(func)
    async def wrapped(*args, **kwargs):
        return await func(*args, **kwargs)

    wrapped.remote = False
    wrapped_asyncgen.remote = False

    if inspect.isasyncgenfunction(func):
        return wrapped_asyncgen
    else:
        return wrapped
