"""Port, IO Device."""

import asyncio
from concert.base import AccessorNotImplementedError, State, check
from concert.coroutines.base import background
from concert.quantities import q
from concert.devices.base import Device


class Signal(Device):

    """Base device for binary signals, e.g. TTL trigger signals and similar."""

    state = State(default='off')

    @background
    @check(source='off', target='on')
    async def on(self):
        """
        on()

        Switch the signal on.
        """
        await self._on()

    @background
    @check(source='on', target='off')
    async def off(self):
        """
        off()

        Switch the signal off.
        """
        await self._off()

    @background
    @check(source='off', target='off')
    async def trigger(self, duration=10 * q.ms):
        """
        trigger(duration=10*q.ms)

        Generate a trigger signal of *duration*.
        """
        await self.on()
        await asyncio.sleep(duration.to(q.s).magnitude)
        await self.off()

    async def _on(self):
        """Implementation."""
        raise NotImplementedError

    async def _off(self):
        """Implementation."""
        raise NotImplementedError


class IO(Device):

    """The IO device consists of ports which can be readable, writable or
    both.
    """

    async def __ainit__(self):
        await super(IO, self).__ainit__()
        self._ports = []

    @property
    def ports(self):
        """Port IDs used by :meth:`.read_port` and :meth:`.write_port`"""
        return self._ports

    def _check(self, port):
        """Check if *port* is a part of the device."""
        if port not in self._ports:
            raise IODeviceError("Port `{}' not found".format(port))

    @background
    async def read_port(self, port):
        """Read a *port*."""
        self._check(port)
        return await self._read_port(port)

    @background
    async def write_port(self, port, value):
        """Write a *value* to the *port*."""
        self._check(port)
        await self._write_port(port, value)

    async def _read_port(self, port):
        """Implementation of reading a *port* from the device."""
        raise AccessorNotImplementedError

    async def _write_port(self, port, value):
        """Implementation of writing a *value* to a *port* on the device."""
        raise AccessorNotImplementedError


class IODeviceError(Exception):

    """Specific exception thrown by IO devices."""

    pass
