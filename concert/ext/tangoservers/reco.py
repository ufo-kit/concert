"""Tango device for online 3D reconstruction from zmq image stream."""
import functools
import multiprocessing
import numpy as np
from tango import DebugIt, CmdArgType, PipeWriteType
from tango.server import command, pipe
from .base import TangoRemoteProcessing
from concert.coroutines.base import async_generate
from concert.ext.ufo import GeneralBackprojectManager, LocalGeneralBackprojectArgs
from concert.networking.base import get_tango_device, ZmqSender
from concert.storage import RemoteDirectoryWalker
from multiprocessing.pool import ThreadPool
from ...config import DISTRIBUTED_TANGO_TIMEOUT

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
async def get_{0}(self):
    arg = await self._args.get_reco_arg("{0}")
    if arg is None:
        arg = self._default['{0}']

    return arg

@DebugIt()
async def set_{0}(self, values):
    try:
        {1}[0]
        # For some reason tango converts normal 'float' to np.float64 which glib doesn't like
        await self._args.set_reco_arg("{0}", [{1}[0](value) for value in values])
    except TypeError:
        await self._args.set_reco_arg("{0}", values)
"""


class TangoOnlineReconstruction(TangoRemoteProcessing):
    """
    Tango device for online 3D reconstruction from zmq image stream.
    """
    _args = LocalGeneralBackprojectArgs()
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

    find_parameter_args = pipe(
        label="find_parameter_args",
        doc="Arguments for finding 3D reconstruction parameters",
        access=PipeWriteType.PIPE_READ_WRITE
    )

    async def init_device(self):
        """Inits device and communciation"""
        await super().init_device()

        self._manager = await GeneralBackprojectManager(
            self._args,
            average_normalization=True
        )
        self._walker = None
        self._sender = None
        self._find_args = None

    @DebugIt()
    @command()
    async def reset_manager(self):
        return self._manager.reset()

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

        walker_device = get_tango_device(tango_remote)
        walker_device.set_timeout_millis(DISTRIBUTED_TANGO_TIMEOUT)
        await walker_device.write_attribute('endpoint', f"{protocol}://{host}:{port}")
        self._walker = await RemoteDirectoryWalker(
            device=walker_device,
            root=root,
            bytes_per_file=2 ** 40,
        )
        if self._sender:
            await self._sender.close()
        self._sender = await ZmqSender(endpoint=f"{protocol}://*:{port}", reliable=True)

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

    @DebugIt()
    @command(dtype_out=int)
    async def get_best_slice_index(self):
        if self._manager.volume is None:
            raise RuntimeError("Volume is empty")

        pool = ThreadPool(processes=max(1, multiprocessing.cpu_count() - 2))
        func = functools.partial(compute_sag_metric, self._manager.volume)
        result = pool.map(func, np.arange(self._manager.volume.shape[0]))

        return np.argmin(result)

    @DebugIt(show_args=True)
    @command(dtype_in=str)
    async def rereconstruct(self, slice_directory):
        await self._reconstruct(cached=True, slice_directory=slice_directory)

    @DebugIt()
    async def write_find_parameter_args(self, args):
        name, blob = args
        self._find_args = dict([(arg["name"], arg["value"]) for arg in blob])
        self.debug_stream("find arguemnts: %s", self._find_args)

    @DebugIt(show_ret=True)
    async def read_find_parameter_args(self):
        if not self._find_args:
            raise RuntimeError("Arguemnts not set")
        return "find_parameter_args", self._find_args

    @DebugIt(show_ret=True)
    @command(dtype_out=float, doc_out="Found rotation axis")
    async def find_parameter(self):
        if not self._find_args:
            raise RuntimeError("Arguemnts not set")

        return (
            await self._manager.find_parameters(
                [self._find_args["parameter"]],
                regions=[self._find_args["region"].astype(float).tolist()],
                metrics=[self._find_args["metric"]],
                store=self._find_args["store"],
                z=self._find_args["z"],
            )
        )[0]

    @DebugIt(show_ret=True)
    @command(dtype_out=(int,))
    def get_volume_shape(self):
        if self._manager.volume is None:
            raise RuntimeError("Volume not available yet")
        return self._manager.volume.shape

    @DebugIt()
    @command(dtype_in=int, dtype_out=(np.float32,))
    def get_slice_x(self, index):
        if self._args.slice_metric:
            raise RuntimeError("x slice accessible only when slice_metric is not specified")
        return self._manager.volume[:, :, index].flatten()

    @DebugIt()
    @command(dtype_in=int, dtype_out=(np.float32,))
    def get_slice_y(self, index):
        if self._args.slice_metric:
            raise RuntimeError("y slice accessible only when slice_metric is not specified")
        return self._manager.volume[:, index, :].flatten()

    @DebugIt()
    @command(dtype_in=int, dtype_out=(np.float32,))
    def get_slice_z(self, index):
        if self._args.slice_metric:
            raise RuntimeError("z slice accessible only when slice_metric is not specified")
        return self._manager.volume[index].flatten()

    @DebugIt()
    @command(dtype_out=(np.float32,))
    def get_volume_line(self):
        """This is called when slice_metric is specified and the result is a line."""
        if not self._args.slice_metric:
            raise RuntimeError("slice_metric must be specified")
        return self._manager.volume

    @DebugIt()
    @command(dtype_out=(str,))
    async def get_z_parameters(self):
        return await self._args.get_reco_arg("z_parameters")

    @DebugIt()
    @command(dtype_out=(str,))
    async def get_slice_metrics(self):
        return await self._args.get_reco_arg("slice_metrics")

    @DebugIt()
    @command(dtype_out=(str,))
    async def get_parameters(self):
        result = []
        for param, doc in await self._args.get_parameters():
            result.append(param)
            result.append(doc)

        return result


def compute_sag_metric(volume, index):
    return np.sum(np.abs(np.gradient(volume[index])))
