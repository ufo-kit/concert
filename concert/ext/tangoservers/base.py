"""Base Tango server for processing zmq data streams."""
import asyncio
import tango
from concert.coroutines.base import start
from concert.networking.base import ZmqReceiver
from tango import InfoIt, DebugIt
from tango.server import Device, attribute, DeviceMeta
from tango.server import AttrWriteType, command


class TangoRemoteProcessing(Device, metaclass=DeviceMeta):
    green_mode = tango.GreenMode.Asyncio
    """Base Tango device for processing zmq streams."""

    endpoint = attribute(
        label="Endpoint",
        dtype=str,
        access=AttrWriteType.READ_WRITE,
        fget="get_endpoint",
        fset="set_endpoint"
    )

    def __init__(self, cl, name):
        self._endpoint = None
        self._receiver = ZmqReceiver()
        self._task = None
        super().__init__(cl, name)

    async def init_device(self):
        """Inits device and communciation."""
        self.info_stream('%s init_device', self.__class__.__name__)
        await super().init_device()
        if self._task and not self._task.done():
            self.debug_stream("Cancelling task: %s", self._task.cancel())
        if self._receiver.endpoint:
            self._receiver.connect(self._receiver.endpoint)
        self.set_state(tango.DevState.STANDBY)

    @DebugIt()
    @command()
    async def teardown(self):
        """Stop receiving data forever."""
        self._receiver.close()

    @DebugIt()
    @command()
    async def reset_connection(self):
        """Stop receiving data forever."""
        if not self._endpoint:
            raise RuntimeError('Endpoint not set')
        self._receiver.close()
        self._receiver.connect(self._endpoint)

    @InfoIt()
    async def get_endpoint(self):
        """Get current endpoint."""
        return self._receiver.endpoint if self._receiver.endpoint else ''

    @InfoIt(show_args=True)
    async def set_endpoint(self, endpoint):
        """Set endpoint."""
        if self._task and not self._task.done():
            raise RuntimeError("Endpoint cannot be set while streaming")
        self._receiver.connect(endpoint)
        self._endpoint = endpoint

    async def _process_stream(self, consumer_coro):
        """Process the data stream by *consumer_coro* and handle state."""
        def callback(task):
            try:
                exc = task.exception()
            except asyncio.CancelledError:
                exc = 'CancelledError'
            self.debug_stream(
                "`%s' done, cancelled: %s, exception: `%s'",
                task._coro.__qualname__,
                task.cancelled(),
                exc
            )
            if exc:
                if exc != 'CancelledError':
                    raise exc
            self.set_state(tango.DevState.STANDBY)

        if self._task and not self._task.done():
            raise RuntimeError("Previous stream still running")
        self._task = start(consumer_coro)
        self._task.add_done_callback(callback)
        self.set_state(tango.DevState.RUNNING)

        try:
            await self._task
        finally:
            self._task = None

    @DebugIt()
    @command()
    async def cancel(self):
        """Stop processing immediately."""
        if self._task:
            self._task.cancel()
