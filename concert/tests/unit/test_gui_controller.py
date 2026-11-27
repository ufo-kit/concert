"""Controller behavior needs no QApplication or Qt dependency."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from concert.base import Parameterizable, Quantity, Parameter, State
from concert.gui.controller import ParameterController, UNREAD
from concert.quantities import q
from concert.tests import TestCase


class TestController(TestCase):
    async def test_public_metadata_and_software_state(self):
        class Device(Parameterizable):
            state = State(default='standby')
            command = Parameter(help='Write command')

            async def _set_command(self, value):
                pass

        device = await Device()
        assert device['state'].readable
        assert await device['state'].get() == 'standby'
        assert not device['command'].readable
        assert device['command'].writable
        assert device['command'].help == 'Write command'

    def make_param(self):
        return SimpleNamespace(get=AsyncMock(return_value=1), set=AsyncMock(),
                               writable=True, locked=False, target_readable=False)

    async def test_request_claims_operation_before_execution(self):
        param = self.make_param()
        controller = ParameterController(param)
        assert controller.value is UNREAD
        task = controller.request('read')
        assert controller.busy
        assert controller.request('write', 2) is None
        await controller.read()
        await task
        param.get.assert_awaited_once()
        param.set.assert_not_awaited()
        assert controller.value == 1

    async def test_shutdown_does_not_cancel_write(self):
        param = self.make_param()
        started, release = asyncio.Event(), asyncio.Event()

        async def setter(value):
            started.set()
            await release.wait()

        param.set.side_effect = setter
        controller = ParameterController(param)
        caller = asyncio.create_task(controller.write(7))
        await started.wait()
        caller.cancel()
        await asyncio.gather(caller, return_exceptions=True)
        assert controller.busy
        shutdown = asyncio.create_task(controller.shutdown())
        await asyncio.sleep(0)
        assert not shutdown.done()
        release.set()
        await shutdown
        assert not controller.busy

    async def test_read_failure_and_recovery(self):
        param = self.make_param()
        controller = ParameterController(param)
        await controller.read()
        param.get.side_effect = RuntimeError('disconnected')
        await controller.read()
        assert str(controller.error) == 'disconnected'
        assert controller.value == 1
        param.get.side_effect = None
        param.get.return_value = None
        await controller.read()
        assert controller.value is None and controller.error is None

    async def test_close_cancels_read(self):
        param = self.make_param()
        release = asyncio.Event()
        param.get.side_effect = lambda: None

        async def getter():
            await release.wait()

        param.get.side_effect = getter
        controller = ParameterController(param)
        controller.request('read')
        await asyncio.sleep(0)
        await controller.shutdown()
        assert controller.closed and not controller.busy
        assert controller.value is UNREAD

    async def test_limits_and_targets(self):
        class Device(Parameterizable):
            position = Quantity(q.mm, lower=0 * q.mm, upper=10 * q.mm)

            async def __ainit__(self):
                await super().__ainit__()
                self.writes = []

            async def _get_position(self):
                return 2 * q.mm

            async def _get_target_position(self):
                return 3 * q.mm

            async def _set_position(self, value):
                self.writes.append(value)

        device = await Device()
        controller = ParameterController(device['position'])
        await controller.read()
        assert controller.target == 3 * q.mm
        await controller.write(20 * q.mm)
        assert controller.error is not None and device.writes == []
        device['position'].lock()
        await controller.write(4 * q.mm)
        assert 'locked' in str(controller.error)
        device['position'].unlock()
        await controller.write(4 * q.mm)
        assert device.writes == [4 * q.mm]

    async def test_limit_update_survives_close(self):
        started, release = asyncio.Event(), asyncio.Event()

        async def upper_setter(value):
            started.set()
            await release.wait()

        class Device(Parameterizable):
            position = Quantity(q.mm, lower=0 * q.mm, upper=10 * q.mm,
                                user_upper_setter=upper_setter)

        device = await Device()
        controller = ParameterController(device['position'])
        controller.request('limits', ('upper', 8 * q.mm))
        await started.wait()
        shutdown = asyncio.create_task(controller.shutdown())
        await asyncio.sleep(0)
        assert not shutdown.done()
        release.set()
        await shutdown
        assert controller.error is None and controller.updated_limit == 'upper'
