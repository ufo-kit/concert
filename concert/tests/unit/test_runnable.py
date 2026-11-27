import asyncio
import json
import tempfile
import unittest

from concert.base import RunnableParameterizable, StateError, TransitionNotAllowed
from concert.coroutines.base import background
from concert.experiments.base import Acquisition, local
from concert.runnable import ManagedRunnable
from concert.storage import DirectoryWalker


class SimpleRunnable(RunnableParameterizable):
    async def __ainit__(self, **kwargs):
        self.started = asyncio.Event()
        self.continue_run = asyncio.Event()
        await super().__ainit__(**kwargs)

    @background
    async def _run(self):
        self.started.set()
        await self.continue_run.wait()


class TestRunnableParameterizable(unittest.IsolatedAsyncioTestCase):
    async def test_acquisition_uses_simple_execution(self):
        @local
        async def produce():
            yield 1

        acquisition = await Acquisition(name="data", producer_corofunc=produce)
        self.assertNotIsInstance(acquisition, ManagedRunnable)
        self.assertEqual([param.name for param in acquisition], ["state"])
        await acquisition.run()
        self.assertEqual(await acquisition.get_state(), "standby")

    async def test_has_only_execution_parameters(self):
        runnable = await SimpleRunnable()
        self.assertEqual([param.name for param in runnable], ["state"])
        self.assertFalse(hasattr(runnable, "walker"))
        self.assertFalse(hasattr(runnable, "prepare"))

    async def test_state_and_repeated_runs(self):
        runnable = await SimpleRunnable()
        self.assertEqual(await runnable.get_state(), "standby")
        task = runnable.run()
        await asyncio.wait_for(runnable.started.wait(), timeout=1)
        self.assertEqual(await runnable.get_state(), "running")
        with self.assertRaises(TransitionNotAllowed):
            await runnable.run()
        runnable.continue_run.set()
        await task
        self.assertEqual(await runnable.get_state(), "standby")
        await runnable.run()
        self.assertEqual(await runnable.get_state(), "standby")

    async def test_cancellation(self):
        runnable = await SimpleRunnable()
        task = runnable.run()
        await asyncio.wait_for(runnable.started.wait(), timeout=1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(await runnable.get_state(), "cancelled")


class LifecycleRunnable(ManagedRunnable):
    async def __ainit__(self, *, blocking_stage=None, failing_stage=None, walker=None, **kwargs):
        self.blocking_stage = blocking_stage
        self.failing_stage = failing_stage
        self.stage_started = asyncio.Event()
        self.continue_stage = asyncio.Event()
        self.run_count = 0
        await super().__ainit__(walker=walker, **kwargs)

    async def _visit_stage(self, stage):
        if stage == self.blocking_stage and not self.stage_started.is_set():
            self.stage_started.set()
            await self.continue_stage.wait()
        if stage == self.failing_stage:
            raise RuntimeError(f"{stage} failed")

    async def early_prepare(self):
        await self._visit_stage("early_prepare")

    async def _prepare_metadata_str(self):
        await self._visit_stage("logging")
        return "{}"

    async def prepare(self):
        await self._visit_stage("prepare")

    @background
    async def _run(self):
        self.run_count += 1
        await self._visit_stage("run")

    async def finish(self):
        await self._visit_stage("finish")

    async def late_finish(self):
        await self._visit_stage("late_finish")


class TestManagedRunnableLifecycle(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_cleans_up_run_directory(self):
        with tempfile.TemporaryDirectory() as root:
            walker = await DirectoryWalker(root=root)
            runnable = await LifecycleRunnable(walker=walker, blocking_stage="run")
            task = runnable.run()
            await asyncio.wait_for(runnable.stage_started.wait(), timeout=1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(await runnable.get_state(), "cancelled")
            self.assertEqual(await walker.get_current(), root)
            self.assertTrue(runnable.ready_to_prepare_next_outer_loop.is_set())
            self.assertEqual(await runnable.get_iteration(), 1)

    async def test_generic_metadata(self):
        runnable = await LifecycleRunnable()
        metadata = json.loads(await ManagedRunnable._prepare_metadata_str(runnable))
        self.assertIn("runnable", metadata)
        self.assertNotIn("experiment", metadata)
        self.assertEqual(metadata["runnable"]["separate_runs"], "True")
        self.assertEqual(await runnable.get_name_fmt(), "run_{:>04}")

    async def test_separate_runs_and_existing_directories(self):
        with tempfile.TemporaryDirectory() as root:
            walker = await DirectoryWalker(root=root)
            await walker.descend("run_0000")
            await walker.ascend()
            runnable = await LifecycleRunnable(walker=walker)
            self.assertEqual(await runnable.get_iteration(), 1)
            await runnable.run()
            self.assertEqual(await runnable.get_current_name(), "run_0001")
            self.assertTrue(await walker.exists("run_0001"))
            self.assertEqual(await walker.get_current(), root)
            self.assertEqual(await runnable.get_iteration(), 2)

    async def test_shared_run_directory(self):
        with tempfile.TemporaryDirectory() as root:
            walker = await DirectoryWalker(root=root)
            runnable = await LifecycleRunnable(walker=walker, separate_runs=False)
            await runnable.run()
            self.assertFalse(await walker.exists("run_0000"))
            self.assertTrue(await walker.exists("runnable_start.json"))
            self.assertEqual(await walker.get_current(), root)

    async def test_readiness_alias(self):
        runnable = await LifecycleRunnable()
        self.assertIs(runnable.ready_to_prepare_next_sample,
                      runnable.ready_to_prepare_next_outer_loop)
        await runnable.run()
        self.assertTrue(runnable.ready_to_prepare_next_outer_loop.is_set())

    async def test_state_covers_complete_lifecycle(self):
        for stage in ("early_prepare", "logging", "prepare", "run", "finish", "late_finish"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as root:
                walker = await DirectoryWalker(root=root)
                runnable = await LifecycleRunnable(blocking_stage=stage, walker=walker)

                first_run = runnable.run()
                await asyncio.wait_for(runnable.stage_started.wait(), timeout=1)
                self.assertEqual(await runnable.get_state(), "running")

                with self.assertRaises(TransitionNotAllowed):
                    await runnable.run()

                runnable.continue_stage.set()
                await first_run
                self.assertEqual(runnable.run_count, 1)
                self.assertEqual(await runnable.get_state(), "standby")

    async def test_failure_in_any_lifecycle_stage_sets_error_state(self):
        for stage in ("early_prepare", "logging", "prepare", "run", "finish", "late_finish"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as root:
                walker = await DirectoryWalker(root=root)
                runnable = await LifecycleRunnable(failing_stage=stage, walker=walker)

                with self.assertRaises((RuntimeError, StateError)):
                    await runnable.run()

                self.assertEqual(await runnable.get_state(), "error")
