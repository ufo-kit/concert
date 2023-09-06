"""
Tango device for image writing of zmq image stream.
"""
import concert.writers
from concert.helpers import PerformanceTracker
from concert.quantities import q
from tango import InfoIt, DebugIt
from PyTango.server import attribute
from PyTango.server import AttrWriteType, command
from .base import TangoRemoteProcessing
from concert.storage import DirectoryWalker

class TangoWriter(TangoRemoteProcessing):
    """
    Tango device for image writing of zmq image stream.
    """

    writer_class = attribute(
        label="Writerclass",
        dtype=str,
        access=AttrWriteType.READ_WRITE,
        fget="get_writer_class",
        fset="set_writer_class"
    )

    dsetname = attribute(
        label="Dsetname",
        dtype=str,
        access=AttrWriteType.READ_WRITE,
        fget="get_dsetname",
        fset="set_dsetname"
    )

    bytes_per_file = attribute(
        label="bytes_per_file",
        dtype=int,
        access=AttrWriteType.READ_WRITE,
        fget="get_bytes_per_file",
        fset="set_bytes_per_file"
    )

    async def init_device(self):
        """Inits device and communciation."""
        await super().init_device()
        self._path = None
        self._writer_class = 'TiffWriter'
        self._dsetname = 'frame_{:>06}.tif'
        self._bytes_per_file = 2 ** 40

    @DebugIt()
    @command(dtype_in=str)
    def set_path(self, path):
        self._path = path

    @DebugIt(show_args=True)
    @command(dtype_in=str)
    async def write_sequence(self, path):
        walker = DirectoryWalker(
            writer=getattr(concert.writers, self._writer_class),
            dsetname=self._dsetname,
            bytes_per_file=self._bytes_per_file,
            root=path
        )
        await self._process_stream(self.consume(walker))

    async def consume(self, walker):
        with PerformanceTracker() as pt:
            total_bytes = await walker.write(self._receiver.subscribe())
            pt.size = total_bytes * q.B

    @InfoIt()
    def get_writer_class(self):
        return self._writer_class

    @InfoIt()
    def set_writer_class(self, writer_class):
        self._writer_class = writer_class

    @InfoIt()
    def get_dsetname(self):
        return self._dsetname

    @InfoIt()
    def set_dsetname(self, dsetname):
        self._dsetname = dsetname

    @InfoIt()
    def get_bytes_per_file(self):
        return self._bytes_per_file

    @InfoIt()
    def set_bytes_per_file(self, bytes_per_file):
        self._bytes_per_file = bytes_per_file
