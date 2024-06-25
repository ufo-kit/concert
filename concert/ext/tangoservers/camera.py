"""
Tango devices for dummy concert cameras sending images over zmq sockets.
"""
import asyncio
import numpy as np
import tango
from concert.base import StateValue
from concert.devices.cameras.dummy import Base, Camera, FileCamera
from concert.networking.base import ZmqSender
from tango import DebugIt, InfoIt
from tango.server import attribute, AttrWriteType, command, Device, DeviceMeta


def _run_asyncio(func, *args, **kwargs):
    # This will break the second the owner of the *func* needs a specific event loop
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(func(*args, **kwargs))
    finally:
        loop.close()


class TangoCamera(Device, metaclass=DeviceMeta):
    green_mode = tango.GreenMode.Asyncio
    """
    Tango device for the dummy concert camera sending images over a zmq socket.
    """

    endpoint = attribute(
        label="Endpoint",
        dtype=str,
        access=AttrWriteType.READ_WRITE,
        fget="get_endpoint",
        fset="set_endpoint"
    )

    send_poison_pill = attribute(
        label="send_poison_pill",
        dtype=bool,
        access=AttrWriteType.READ_WRITE,
        fget="get_send_poison_pill",
        fset="set_send_poison_pill"
    )

    def __init__(self, cl, name):
        self._endpoint = None
        self._send_poison_pill = True
        self._sender = ZmqSender()
        self._params_initialized = False
        self.camera = None
        super().__init__(cl, name)

    async def init_device(self):
        if self.camera is None:
            self.camera = await Base()
        self.info_stream('%s init_device', self.__class__.__name__)
        await super().init_device()
        self._index = 0
        self._stop_streaming_requested = False
        if not self._params_initialized:
            for param in self.camera:
                if param.name == "zmq_options":
                    continue
                self.debug_stream("Adding `%s' attribute", param.name)
                default = await param.get()
                write_type = AttrWriteType.READ_WRITE if param.writable else AttrWriteType.READ
                has_unit = hasattr(param, 'unit')
                if has_unit:
                    default = default.magnitude
                if not isinstance(param, StateValue):
                    attr = tango.Attr(
                        param.name,
                        tango.utils.TO_TANGO_TYPE[type(default)],
                        write_type
                    )
                    self.add_attribute(
                        attr,
                        r_meth=self.dynamic_getter,
                        w_meth=self.dynamic_setter if param.writable else None
                    )
            self._params_initialized = True

    def dynamic_getter(self, attr):
        # If this were async def Tango would complain about it never being awaited
        self.debug_stream("reading attribute %s", attr.get_name())
        param = self._camera[attr.get_name()]
        value = _run_asyncio(param.get)
        if hasattr(param, 'unit'):
            value = value.magnitude
        attr.set_value(value)

    def dynamic_setter(self, attr):
        self.debug_stream("writting attribute %s", attr.get_name())
        param = self._camera[attr.get_name()]
        value = attr.get_write_value()
        if hasattr(param, 'unit'):
            value = value * param.unit
        _run_asyncio(param.set, value)

    @DebugIt()
    @command()
    async def teardown(self):
        """Stop receiving data forever."""
        self._sender.close()

    @DebugIt()
    @command()
    async def reset_connection(self):
        """Stop receiving data forever."""
        if not self._endpoint:
            raise RuntimeError('Endpoint not set')
        self._sender.close()
        self._sender.connect(self._endpoint)

    @InfoIt()
    async def get_endpoint(self):
        """Get current endpoint."""
        return self._sender.endpoint if self._sender.endpoint else ''

    @DebugIt(show_ret=True)
    @command(dtype_out=str)
    async def get_camera_state(self):
        if self._camera:
            return await self._camera.get_state()
        else:
            return 'uninitialized'

    @InfoIt(show_args=True)
    async def set_endpoint(self, endpoint):
        """Set endpoint."""
        self._sender.connect(endpoint)
        self._endpoint = endpoint

    @DebugIt(show_ret=True)
    async def get_send_poison_pill(self):
        return self._send_poison_pill

    @DebugIt(show_args=True)
    async def set_send_poison_pill(self, value):
        self._send_poison_pill = value

    async def _start_recording(self):
        if not self._camera:
            raise RuntimeError('Camera has not been initialized')
        self._index = 0
        await self._camera.start_recording()

    @DebugIt()
    @command()
    async def stop_recording(self):
        if not self._camera:
            raise RuntimeError('Camera has not been initialized')

        await self._camera.stop_recording()

    @DebugIt()
    @command()
    async def cancel_grabbing(self):
        self._stop_streaming_requested = True

    @command(dtype_in=int)
    async def grab(self, num):
        send_poison_pill = self._send_poison_pill
        for i in range(num):
            if self._stop_streaming_requested:
                # Immediately reset
                send_poison_pill = True
                self._stop_streaming_requested = False
                self.debug_stream('Stop stream upon request')
                break
            self.debug_stream(
                'Grab %d/%d (%d from start_recording)',
                i + 1,
                num,
                self._index + 1
            )
            # If ROI is applied, zmq would complain about non-contiguous memory, this is not at
            # all supposed to be high-performance camera, so an additional copy doesn't matter.
            await self._sender.send_image(np.copy(await self._camera.grab()))
            self._index += 1

        if send_poison_pill:
            self.debug_stream('Sending poisson pill')
            await self._sender.send_image(None)


class TangoDummyCamera(TangoCamera):

    """Remote counterpart of :class:`concert.devices.cameras.dummy.Camera`."""

    simulate = attribute(
        label="Simulate",
        dtype=bool,
        access=AttrWriteType.READ_WRITE,
        fget="get_simulate",
        fset="set_simulate"
    )

    background = attribute(
        label="Background",
        max_dim_x=2 ** 15,
        max_dim_y=2 ** 15,
        dtype=((np.uint16,),),
        access=AttrWriteType.WRITE,
        fset="set_background"
    )

    async def init_device(self):
        self.camera = await Camera()
        await super().init_device()

    @command()
    async def start_recording(self):
        await super()._start_recording()

    @DebugIt()
    async def get_simulate(self):
        return await self._camera.get_simulate()

    @DebugIt(show_args=True)
    async def set_simulate(self, simulate):
        await self._camera.set_simulate(simulate)

    @DebugIt()
    async def set_background(self, background):
        await self._camera.set_background(background)


class TangoFileCamera(TangoCamera):

    """Remote counterpart of :class:`concert.devices.cameras.dummy.FileCamera`."""

    pattern = attribute(
        label="Pattern",
        dtype=str,
        access=AttrWriteType.READ_WRITE,
        fget="get_pattern",
        fset="set_pattern"
    )

    reset_on_start = attribute(
        label="Reset_on_start",
        dtype=bool,
        access=AttrWriteType.READ_WRITE,
        fget="get_reset_on_start",
        fset="set_reset_on_start"
    )

    start_index = attribute(
        label="Start_index",
        dtype=int,
        access=AttrWriteType.READ_WRITE,
        fget="get_start_index",
        fset="set_start_index"
    )

    def __init__(self, cl, name):
        self._pattern = ''
        self._reset_on_start = True
        self._start_index = 0
        self._camera = None
        super().__init__(cl, name)

    async def init_device(self):
        await super().init_device()
        if self._pattern:
            self._camera = await FileCamera(
                self._pattern,
                reset_on_start=self._reset_on_start,
                start_index=self._start_index
            )

    @InfoIt(show_args=True)
    async def set_pattern(self, pattern):
        if self._camera and await self._camera.get_state() != 'standby':
            raise RuntimeError("Pattern can be changed only in `standby' state")
        self._pattern = pattern
        if not self._camera:
            await self.init_device()
        else:
            await self._camera.set_pattern(pattern)

    @InfoIt()
    async def get_pattern(self):
        return self._pattern

    @InfoIt(show_args=True)
    async def set_reset_on_start(self, reset_on_start):
        self._reset_on_start = reset_on_start

    @InfoIt()
    async def get_reset_on_start(self):
        return self._reset_on_start

    @InfoIt(show_args=True)
    async def set_start_index(self, start_index):
        self._start_index = start_index

    @InfoIt()
    async def get_start_index(self):
        return self._start_index

    @DebugIt()
    @command()
    async def start_recording(self):
        await super()._start_recording()
        if self._camera.reset_on_start:
            self._camera.index = 0
