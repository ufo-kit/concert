"""Tango device for online 3D reconstruction from zmq image stream."""
import numpy as np
from tango import DebugIt, CmdArgType
from tango.server import command
from .base import TangoRemoteProcessing
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

    async def init_device(self):
        """Inits device and communciation"""
        await super().init_device()

        self._manager = await GeneralBackprojectManager(
            self._args,
            average_normalization=True
        )
        self._walker = None
        self._sender = None

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

    @DebugIt(show_args=True)
    @command(dtype_in=str)
    async def reconstruct(self, slice_directory):
        await self._process_stream(self._manager.backproject(self._receiver.subscribe()))
        if slice_directory:
            f = self._walker.write_sequence(path=slice_directory)
            for image in self._manager.volume:
                await self._sender.send_image(image)
            # Poison pill
            await self._sender.send_image(None)

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
