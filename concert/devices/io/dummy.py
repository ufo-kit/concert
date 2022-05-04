"""Dummy IO"""
from concert.base import transition
from concert.devices.io import base


class IO(base.IO):

    """Dummy I/O device implementation."""

    async def __ainit__(self, port_value=0):
        await super(IO, self).__ainit__()
        self._ports = {0: port_value}

    async def _read_port(self, port):
        return self._ports[port]

    async def _write_port(self, port, value):
        self._ports[port] = value


class Signal(base.Signal):

    """Dummy signal device."""

    @transition(target='on')
    async def _on(self):
        pass

    @transition(target='off')
    async def _off(self):
        pass
