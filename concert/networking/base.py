"""Connection protocols for network communication."""
import asyncio
import logging
from concert.quantities import q
from concert.config import AIODEBUG


LOG = logging.getLogger(__name__)


class SocketConnection(object):

    """A two-way socket connection. *return_sequence* is a string appended
    after every command indicating the end of it, the default value
    is a newline (\\n).
    """

    def __init__(self, host, port, return_sequence="\n"):
        self._peer = (host, port)
        self._reader = None
        self._writer = None
        self.lock = asyncio.Lock()
        self.return_sequence = return_sequence

    async def __aenter__(self):
        LOG.log(AIODEBUG, 'Socket connection enter')
        await self.connect()

        return self

    async def __aexit__(self, exc_type, exc, tb):
        LOG.log(AIODEBUG, 'Socket connection exit')
        await self.close()

    async def connect(self):
        """Open connection."""
        self._reader, self._writer = await asyncio.open_connection(*self._peer)

    async def close(self):
        """Close connection."""
        self._writer.close()
        await self._writer.wait_closed()

    async def send(self, data):
        """
        Send *data* to the peer. The return sequence characters
        are appended to the data before it is sent.
        """
        if not self._reader:
            await self.connect()

        LOG.debug('Sending {0}'.format(data))
        data += self.return_sequence
        self._writer.write(data.encode('ascii'))
        await self._writer.drain()

    async def recv(self, num=1024):
        """
        Read *num* bytes from the socket. The result is first stripped from the trailing return
        sequence characters and then returned.
        """
        if not self._reader:
            await self.connect()

        result = (await self._reader.read(num)).decode('ascii')
        if result.endswith(self.return_sequence):
            # Strip the command-ending character
            result = result.rstrip(self.return_sequence)
        LOG.debug('Received {0}'.format(result))
        return result

    async def execute(self, data, num=1024):
        """Execute command and wait for response (coroutine-safe, not thread-safe). Read *num* bytes
        from the socket.
        """
        async with self.lock:
            await self.send(data)
            result = await self.recv(num=num)

        return result


def get_tango_device(uri, peer=None, timeout=10 * q.s):
    """
    Get a Tango device by specifying its *uri*. If *peer* is given change the tango_host specifying
    which database to connect to. Format is host:port as a string. *timeout* sets the device's
    general timeout. It is converted to milliseconds, converted to integer and then the tango
    device's `set_timout_millis` is called with the converted integer value.
    """
    import IPython
    import tango

    if peer is not None:
        uri = f"{peer}/{uri}"

    executor = None
    if IPython.version_info >= (8, 0):
        from IPython.core.async_helpers import get_asyncio_loop
        ipython_loop = get_asyncio_loop()
        from tango import asyncio_executor
        executor = asyncio_executor.AsyncioExecutor(loop=ipython_loop)

    device = tango.DeviceProxy(
        uri, green_mode=tango.GreenMode.Asyncio, asyncio_executor=executor
    )
    device.set_timeout_millis(int(timeout.to(q.ms).magnitude))

    return device
