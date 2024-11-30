"""Tango device for online 3D reconstruction from zmq image stream."""
import asyncio
import os
from typing import List
import numpy as np
try:
    from numpy.typing import ArrayLike
except ModuleNotFoundError:
    from numpy import ndarray as ArrayLike
from tango import DebugIt, CmdArgType, EventType, EventData, GreenMode
from tango.server import command
from tango.asyncio import DeviceProxy
from .base import TangoRemoteProcessing
from concert.coroutines.base import async_generate
from concert.ext.ufo import GeneralBackprojectManager, GeneralBackprojectArgs
from concert.quantities import q
from concert.networking.base import get_tango_device, ZmqSender
from concert.storage import RemoteDirectoryWalker


MAX_DIM = 100000


# "Dynamic" creation of attributes, is there really not a better way? (Attr doesn't work with
# arrays)
FMT = """
# Import here so that flake8 is happy
from tango.server import attribute
from tango.server import AttrWriteType

{0} = attribute(
    label="{0}",
    dtype={1},
    max_dim_x=MAX_DIM,
    access=AttrWriteType.READ_WRITE,
    fget="get_{0}",
    fset="set_{0}"
)

@DebugIt()
def get_{0}(self):
    arg = self._args.{0}
    if arg is None:
        arg = self._default['{0}']

    return arg

@DebugIt()
@command(dtype_in={1})
def set_{0}(self, values):
    try:
        {1}[0]
        # For some reason tango converts normal 'float' to np.float64 which glib doesn't like
        self._args.{0} = [{1}[0](value) for value in values]
    except TypeError:
        self._args.{0} = values
"""


class TangoOnlineReconstruction(TangoRemoteProcessing):
    """
    Tango device for online 3D reconstruction from zmq image stream.
    """
    _args = GeneralBackprojectArgs()
    _default = {}

    for arg, settings in _args.parameters.items():
        dtype = settings.get('type')
        default = settings.get('default')
        if arg == 'gpus':
            from gi.repository import Ufo
            res = Ufo.Resources()
            setattr(_args, arg, [i for i in range(len(res.get_gpu_nodes()))])
            exec(FMT.format(arg, '(int,)'))
        elif settings.get('action') == 'store_true':
            exec(FMT.format(arg, 'bool'))
        elif 'choices' in settings:
            # A selection
            exec(FMT.format(arg, 'str'))
        elif dtype.__name__ == 'split_values':
            # a list
            exec(FMT.format(arg, f"({dtype.dtype.__name__},)"))
            # If None is set in tofu config, it is incompatible with the actual data type and would
            # cause tango error, otherwise use the tofu config default
            _default[arg] = [0.0] if default is None else default
        elif dtype.__name__ == 'check':
            # restrict_value from tofu.util
            exec(FMT.format(arg, dtype.dtype.__name__))
            # If None is set in tofu config, it is incompatible with the actual data type and would
            # cause tango error, thus use the lower limit, otherwise use the tofu config default
            _default[arg] = dtype.limits[0] if default is None else default
        else:
            # Just an ordinary data type
            exec(FMT.format(arg, dtype.__name__))
            _default[arg] = '' if dtype == str else dtype(0)

    # Do not infect the class with temp variables
    del arg, dtype, settings

    _rae_device: DeviceProxy
    _rae_subscription: int
    _slice_directory: str
    _cached: bool
    _found_rotation_axis: asyncio.Event

    async def init_device(self):
        """Inits device and communciation"""
        await super().init_device()
        # Rotation axis event signals GeneralBackprojectManager to start backprojector. We trigger
        # this event from the callback response upon having required parameters estimated.
        self._found_rotation_axis = asyncio.Event()
        self._manager = await GeneralBackprojectManager(
            self._args,
            average_normalization=True,
            found_rotation_axis=self._found_rotation_axis
        )
        self._walker = None
        self._sender = None

    @DebugIt()
    @command(
        dtype_in=str,
        doc_in="absolute path of the directory to write online slices"
    )
    async def set_slice_directory(self, slice_directory: str) -> None:
        self._slice_directory = slice_directory

    @DebugIt()
    @command(
        dtype_in=bool,
        doc_in="whether to reconstruct from cache"
    )
    async def set_cached(self, cached: bool) -> None:
        self._cached = cached

    @DebugIt()
    @command(
        dtype_in=(str,),
        doc_in="device proxy reference for axis of rotation estimator"
    )
    async def register_rotation_axis_feedback(self, args: Tuple[str, str]) -> None:
        """
        Subscribes for event concerning axis of rotation estimation.
        Arguments include port number and domain namespace for rotation axis estimator device.
        """
        # Get the device proxy reference for RotationAxisEstimator device and subscribe for event
        self._rae_device = await DeviceProxy(f"{os.uname()[1]}:{args[0]}/{args[1]}#dbase=no")
        self._rae_subscription = await self._rae_device.subscribe_event(
                "axis_of_rotation", EventType.USER_EVENT, self._on_axis_of_rotation,
                green_mode=GreenMode.Asyncio)

    def _on_axis_of_rotation(self, event: EventData) -> None:
        """
        Defines the callback function upon receiving estimated center of rotation by the QA
        device
        """
        # We wrap the implementation within exception handling to catch transparent issues in Tango
        # and reraise.
        try:
            if event.attr_value.value is not None:
                extracted_arr: ArrayLike = np.vectorize(np.float_)(event.attr_value.value)
                if not np.all(np.isclose(extracted_arr, np.zeros((3,)), atol=1e-5)):
                    self.info_stream("%s: received: %s", self.__class__.__name__, extracted_arr)
                    estm_center, estm_axis_angle_y, estm_axis_angle_x = extracted_arr.tolist()
                    self._manager.args.center_position_x = [estm_center]
                    # TODO: Determine what is the correct way to deal with angular corrections.
                    self._manager.args.axis_angle_y = [estm_axis_angle_y]
                    self._manager.args.axis_angle_x = [estm_axis_angle_x]
                    self._found_rotation_axis.set()
        except Exception as err:
            self.error_stream("%s: encountered error: %s", self.__class__.__name__, str(err))
            raise

    @DebugIt()
    @command()
    async def reset_manager(self):
        return await self._manager.reset()

    @DebugIt()
    @command(dtype_out=str)
    async def get_manager_state(self):
        return await self._manager.get_state()

    @DebugIt()
    @command()
    async def update_darks(self):
        await self._process_stream(self._manager.update_darks(self._receiver.subscribe()))

    @DebugIt()
    @command()
    async def update_flats(self):
        await self._process_stream(self._manager.update_flats(self._receiver.subscribe()))

    @DebugIt()
    @command(
        dtype_in=CmdArgType.DevVarStringArray,
        doc_in="1. walker's tango server URI, "
               "2. root path, "
               "3. protocol, "
               "4. host, "
               "5. port "
    )
    async def setup_walker(self, args):
        """Slice walker for writing slices remotely."""
        tango_remote = args[0]
        root = args[1]
        protocol = args[2]
        host = args[3]
        port = args[4]

        walker_device = get_tango_device(tango_remote, timeout=1000 * q.s)
        await walker_device.write_attribute('endpoint', f"{protocol}://{host}:{port}")
        self._walker = await RemoteDirectoryWalker(
            device=walker_device,
            root=root,
            bytes_per_file=2 ** 40,
        )
        if self._sender:
            self._sender.close()
        self._sender = ZmqSender(endpoint=f"{protocol}://*:{port}", reliable=True)

    async def _reconstruct(self, cached=False, slice_directory=""):
        if cached:
            await self._manager.backproject(async_generate(self._manager.projections))
        else:
            await self._process_stream(self._manager.backproject(self._receiver.subscribe()))
        if slice_directory:
            f = self._walker.write_sequence(name=slice_directory)
            for image in self._manager.volume:
                await self._sender.send_image(image)
            # Poison pill
            await self._sender.send_image(None)

    @DebugIt(show_args=True)
    @command(dtype_in=str)
    async def reconstruct(self, slice_directory):
        await self._reconstruct(cached=False, slice_directory=slice_directory)

    @DebugIt(show_args=True)
    @command(dtype_in=str)
    async def rereconstruct(self, slice_directory):
        await self._reconstruct(cached=True, slice_directory=slice_directory)

    @DebugIt(show_args=True, show_ret=True)
    @command(
        dtype_in=(float,),
        doc_in="1. region start (float), "
               "2. region end (float), "
               "3. region step (float), "
               "4. z position (int), "
               "5. store (bool) ",
        dtype_out=float,
        doc_out="Found rotation axis"
    )
    async def find_axis(self, args):
        region = [float(args[i]) for i in range(3)]
        z = int(args[3])
        store = bool(args[4])
        return (
            await self._manager.find_parameters(
                ["center-position-x"],
                regions=[region],
                z=z,
                store=store
            )
        )[0]

    @DebugIt(show_ret=True)
    @command(dtype_out=(int,))
    def get_volume_shape(self):
        return self._manager.volume.shape

    @DebugIt()
    @command(dtype_in=int, dtype_out=(np.float32,))
    def get_slice_x(self, index):
        return self._manager.volume[:, :, index].flatten()

    @DebugIt()
    @command(dtype_in=int, dtype_out=(np.float32,))
    def get_slice_y(self, index):
        return self._manager.volume[:, index, :].flatten()

    @DebugIt()
    @command(dtype_in=int, dtype_out=(np.float32,))
    def get_slice_z(self, index):
        return self._manager.volume[index].flatten()
