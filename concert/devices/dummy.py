"""Dummy"""
import asyncio
import logging
from concert.base import check, Parameter, Quantity, Selection, State, transition
from concert.coroutines.base import background
from concert.config import AIODEBUG
from concert.devices.base import Device
from concert.quantities import q


LOG = logging.getLogger(__name__)


async def set_evalue(instance, value):
    instance._value = value


async def get_evalue(instance):
    return instance._value


class DummyDevice(Device):

    """A dummy device."""

    position = Quantity(unit=q.mm)
    sleep_time = Quantity(unit=q.s)
    # Value with a check-decorated setter before it is bound to instance, so still a function
    value = Parameter()
    # value with check wrapping the bound method
    cvalue = Parameter(check=check(source='*', target='standby'))
    # Value set elsewhere
    evalue = Parameter(fset=set_evalue, fget=get_evalue, check=check(source='*', target='standby'))
    slow = Parameter()
    state = State(default='standby')

    async def __ainit__(self, slow=None):
        await super(DummyDevice, self).__ainit__()
        self._position = 1 * q.mm
        self._value = 0
        self._slow = slow
        self._sleep_time = 0.5 * q.s

    async def _get_sleep_time(self):
        return self._sleep_time

    async def _set_sleep_time(self, value):
        self._sleep_time = value

    async def _get_position(self):
        return self._position

    async def _set_position(self, value):
        self._position = value

    async def _get_slow(self):
        try:
            LOG.log(AIODEBUG, 'get slow start %s', self._slow)
            await asyncio.sleep((await self.get_sleep_time()).magnitude)
            LOG.log(AIODEBUG, 'get slow finish %s', self._slow)
            return self._slow
        except asyncio.CancelledError:
            LOG.log(AIODEBUG, 'get slow cancelled %s', self._slow)
            raise
        except KeyboardInterrupt:
            # do not scream
            LOG.debug("KeyboardInterrupt caught while getting")

    async def _set_slow(self, value):
        try:
            LOG.log(AIODEBUG, 'set slow start %s', value)
            await asyncio.sleep((await self.get_sleep_time()).magnitude)
            LOG.log(AIODEBUG, 'set slow finish %s', value)
            self._slow = value
        except asyncio.CancelledError:
            LOG.log(AIODEBUG, 'set slow cancelled %s', value)
            raise

    async def _get_value(self):
        """Get the real value."""
        return self._value

    async def _get_target_value(self):
        """Get the real value."""
        return self._value + 1

    @check(source='standby', target=['standby', 'hard-limit'])
    @transition(immediate='moving', target='standby')
    async def _set_value(self, value):
        """The real value setter."""
        self._value = value

    async def _get_cvalue(self):
        """The real value setter."""
        return self._value

    async def _set_cvalue(self, value):
        """The real value setter."""
        self._value = value

    @background
    async def do_nothing(self, value=None):
        """Do nothing. For testing task canellation."""
        await self._do_nothing(value=value)

    async def _do_nothing(self, value=None):
        LOG.log(AIODEBUG, f'Start doing nothing: {value}')
        try:
            await asyncio.sleep(1)
            LOG.log(AIODEBUG, f'Stop doing nothing: {value}')
            return value
        except asyncio.CancelledError:
            LOG.log(AIODEBUG, f'Doing nothing cancelled: {value}')
            raise

    async def _emergency_stop(self):
        LOG.debug('Emergency stop on a dummy device')
        await asyncio.sleep(0.5)
        self._state_value = 'aborted'


class SelectionDevice(Device):

    """A dummy device with a selection."""

    selection = Selection(list(range(3)))

    async def __ainit__(self):
        await super(SelectionDevice, self).__ainit__()
        self._selection = 0

    async def _get_selection(self):
        return self._selection

    async def _set_selection(self, selection):
        self._selection = selection
