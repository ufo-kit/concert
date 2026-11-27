"""Managed execution with per-run storage, logging, and lifecycle hooks."""

import asyncio
import json
import logging
import time

from concert.base import Parameter, RunnableParameterizable, Selection, StateError, TargetAccessError
from concert.helpers import get_basename


LOG = logging.getLogger(__name__)


class ManagedRunnable(RunnableParameterizable):
    """A runnable with preparation, finalization, and optional recorded output.

    With a *walker*, each run can have its own directory, log, and device metadata.
    ``state`` covers the entire lifecycle, including preparation and cleanup.
    """

    separate_runs = Parameter(help="Store each run in a separate directory")
    name_fmt = Parameter(help="Directory name format for each run")
    iteration = Parameter(help="Index of the next run")
    current_name = Parameter(help="Name of the current run")
    log_file_prefix = Parameter()
    log_level = Selection(['critical', 'error', 'warning', 'info', 'debug'])
    log_devices_at_start = Parameter()
    log_devices_at_finish = Parameter()

    async def __ainit__(self, *, walker=None, separate_runs=True,
                        name_fmt='run_{:>04}', **kwargs):
        self._log_file_prefix = None
        self.log = LOG
        self._devices_to_log = {}
        self._devices_to_log_optional = {}
        self._log_devices_at_start = None
        self._log_devices_at_finish = None
        self.ready_to_prepare_next_outer_loop = asyncio.Event()

        # Retained for compatibility with existing experiment subclasses.
        self.ready_to_prepare_next_sample = self.ready_to_prepare_next_outer_loop
        await super().__ainit__(**kwargs)
        self.walker = walker
        await self.set_log_file_prefix("runnable")
        await self.set_log_devices_at_start(True)
        await self.set_log_devices_at_finish(True)
        self._separate_runs = separate_runs
        self._name_fmt = name_fmt
        self._current_name = ""
        self._run_iteration = 0

        if separate_runs and walker:
            # Find an unused run directory rather than overwrite existing data.
            while await self.walker.exists(self._name_fmt.format(self._run_iteration)):
                self._run_iteration += 1

    async def _get_iteration(self):
        return self._run_iteration

    async def _set_iteration(self, iteration):
        self._run_iteration = iteration

    async def _get_current_name(self):
        return self._current_name

    async def _get_separate_runs(self):
        return self._separate_runs

    async def _set_separate_runs(self, separate_runs):
        self._separate_runs = separate_runs

    async def _get_name_fmt(self):
        return self._name_fmt

    async def _set_name_fmt(self, fmt):
        self._name_fmt = fmt

    async def _set_log_devices_at_start(self, log):
        self._log_devices_at_start = bool(log)

    async def _get_log_devices_at_start(self):
        return self._log_devices_at_start

    async def _set_log_devices_at_finish(self, log):
        self._log_devices_at_finish = bool(log)

    async def _get_log_devices_at_finish(self):
        return self._log_devices_at_finish

    async def _get_log_level(self):
        return logging.getLevelName(self.log.getEffectiveLevel()).lower()

    async def _set_log_level(self, level):
        self.log.setLevel(level.upper())

    async def early_prepare(self):
        """Configure the runnable before logging starts."""
        pass

    async def prepare(self):
        """Prepare the operation after initial metadata has been recorded."""
        pass

    async def finish(self):
        """Finalize the operation before recording final metadata."""
        pass

    async def late_finish(self):
        """Finalize the run after recording final metadata."""
        pass

    def add_device_to_log(self, name: str, device: 'concert.devices.base.Device', optional=False):
        """Add a device to the metadata snapshots.

        If *optional* is True, metadata read failures are logged rather than raised.
        """
        if optional:
            self._devices_to_log_optional[name] = device
        else:
            self._devices_to_log[name] = device

    async def _prepare_metadata_str(self) -> str:
        """Serialize runnable parameters and registered device metadata."""
        async def get_parameter_data(parameter):
            param_data: dict[str, object] = {"value": str(await parameter.get())}
            try:
                param_data["target"] = str(await parameter.get_target())
            except TargetAccessError:
                pass

            try:
                param_data["limits"] = {
                    "lower": {
                        "effective": str(await parameter.get_lower()),
                        "user": str(await parameter.get_lower_user()),
                        "external": str(await parameter.get_lower_external()),
                    },
                    "upper": {
                        "effective": str(await parameter.get_upper()),
                        "user": str(await parameter.get_upper_user()),
                        "external": str(await parameter.get_upper_external()),
                    },
                }
            except AttributeError:
                pass
            return param_data

        metadata = {}
        runnable_params = {}
        for param in self:
            runnable_params[param.name] = str(await param.get())
        metadata["runnable"] = runnable_params
        for name, device in self._devices_to_log.items():
            device_data = {}
            for param in device:
                device_data[param.name] = await get_parameter_data(param)
            metadata[name] = device_data

        for name, device in self._devices_to_log_optional.items():
            device_data = {}
            for param in device:
                try:
                    device_data[param.name] = await get_parameter_data(param)
                except Exception as e:
                    self.log.info(f"Error while logging optional device {name}")
                    self.log.info(e)
            metadata[name] = device_data
        return json.dumps(metadata, indent=4)

    async def _get_log_file_prefix(self):
        return self._log_file_prefix

    async def _set_log_file_prefix(self, prefix: str):
        self._log_file_prefix = str(prefix)

    def _execute(self):
        return asyncio.ensure_future(self._run_lifecycle())

    async def _run_lifecycle(self):
        self.ready_to_prepare_next_outer_loop.clear()
        start_time = time.time()
        handler = None
        descended = False
        lifecycle_started = False
        finalization_error = None
        iteration = await self.get_iteration()
        separate_runs = await self.get_separate_runs()

        try:
            await self.early_prepare()
            if self.walker:
                if separate_runs:
                    await self.walker.descend((await self.get_name_fmt()).format(iteration))
                    descended = True
                if await self.walker.exists():
                    handler = await self.walker.register_logger(
                        logger_name=self.__class__.__name__,
                        log_level=logging.NOTSET,
                        file_name=f"{await self.get_log_file_prefix()}.log"
                    )
                    self.log.addHandler(handler)
                    if await self.get_log_devices_at_start():
                        log_metadata = await self._prepare_metadata_str()
                        await self.walker.log_to_json(
                            payload=log_metadata,
                            filename=f"{await self.get_log_file_prefix()}_start.json"
                        )
                self._current_name = get_basename(await self.walker.get_current())
            self.log.info(await self.info_table)
            for name, device in self._devices_to_log.items():
                self.log.info(f"Device {name}:")
                self.log.info(await device.info_table)
            LOG.debug('Run with iteration %d start', iteration)

            lifecycle_started = True
            await self.prepare()
            await self._run()
        finally:
            if lifecycle_started:
                try:
                    await self.finish()
                    if self.walker:
                        if await self.get_log_devices_at_finish():
                            log_metadata = await self._prepare_metadata_str()
                            await self.walker.log_to_json(
                                payload=log_metadata,
                                filename=f"{await self.get_log_file_prefix()}_finish.json"
                            )
                    await self.late_finish()
                except Exception as e:
                    LOG.warning(f"Error `{e}' while finalizing run().")
                    finalization_error = StateError('error', msg=str(e))
            try:
                self.ready_to_prepare_next_outer_loop.set()
                if descended:
                    await self.walker.ascend()
                LOG.debug('Run iteration %d duration: %.2f s',
                          iteration, time.time() - start_time)
                if handler:
                    self.log.removeHandler(handler)
                    await handler.aclose()
                if lifecycle_started:
                    await self.set_iteration(iteration + 1)
            except Exception as e:
                LOG.warning(f"Error `{e}' while cleaning up run().")
                if finalization_error is None:
                    finalization_error = StateError('error', msg=str(e))
            if finalization_error is not None:
                raise finalization_error
