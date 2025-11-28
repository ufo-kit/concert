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

    receiver_reliable = attribute(
        label="Is ZMQ receiver reliable or not",
        dtype=bool,
        access=AttrWriteType.READ_WRITE,
        fget="get_receiver_reliable",
        fset="set_receiver_reliable"
    )

    receiver_rcvhwm = attribute(
        label="Receive high water mark for receiver",
        dtype=int,
        access=AttrWriteType.READ_WRITE,
        fget="get_receiver_rcvhwm",
        fset="set_receiver_rcvhwm"
    )

    def __init__(self, cl, name):
        self._endpoint = ""
        self._receiver_reliable = True
        self._receiver_rcvhm = 0
        self._receiver = None
        self._task = None
        super().__init__(cl, name)

    async def init_device(self):
        """Inits device and communciation."""
        self.info_stream('%s init_device', self.__class__.__name__)
        await super().init_device()
        if self._task and not self._task.done():
            self.debug_stream("Cancelling task: %s", self._task.cancel())
        if self._endpoint:
            await self._create_and_connect_receiver()
        self.set_state(tango.DevState.STANDBY)

    @DebugIt()
    async def _create_and_connect_receiver(self):
        if not self._receiver:
            self._receiver = await ZmqReceiver(
                endpoint=self._endpoint,
                reliable=self._receiver_reliable,
                rcvhwm=self._receiver_rcvhm
            )

        await self._receiver.connect(self._endpoint)

    @DebugIt()
    @command()
    async def connect_endpoint(self):
        """Connect to the zmq endpoint."""
        if not self._endpoint:
            raise RuntimeError("Endpoint not set")
        await self._create_and_connect_receiver()

    @DebugIt()
    @command()
    async def disconnect_endpoint(self):
        """Disconnect from the zmq endpoint."""
        if self._receiver:
            await self._receiver.close()
        self._receiver = None

    @DebugIt()
    @command()
    async def reset_connection(self):
        """Stop receiving data forever."""
        if not self._endpoint:
            raise RuntimeError('Endpoint not set')
        await self._receiver.close()
        self._receiver = None
        await self._create_and_connect_receiver()

    @InfoIt()
    async def get_endpoint(self):
        """Get current endpoint."""
        return self._endpoint

    @InfoIt(show_args=True)
    async def set_endpoint(self, endpoint):
        """Set endpoint."""
        if self._task and not self._task.done():
            raise RuntimeError("Endpoint cannot be set while streaming")
        self._endpoint = endpoint

    @InfoIt()
    async def get_receiver_reliable(self):
        """Get if ZMQ receiver is reliable or not."""
        return self._receiver_reliable

    @InfoIt(show_args=True)
    async def set_receiver_reliable(self, reliable):
        """Set if ZMQ receiver is reliable or not."""
        if self._task and not self._task.done():
            raise RuntimeError("Receiver options cannot be set while streaming")
        self._receiver_reliable = reliable

    @InfoIt()
    async def get_receiver_rcvhwm(self):
        """Get receiver high water mark."""
        return self._receiver_rcvhm

    @InfoIt(show_args=True)
    async def set_receiver_rcvhwm(self, rcvhwm):
        """Set receiver high water mark."""
        if self._task and not self._task.done():
            raise RuntimeError("Receiver options cannot be set while streaming")
        self._receiver_rcvhm = rcvhwm

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

        if not self._receiver:
            await self._create_and_connect_receiver()

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
