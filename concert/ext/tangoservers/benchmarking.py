"""
Tango server for benchmarking zmq transfers.
"""
from concert.helpers import PerformanceTracker
from concert.quantities import q
from tango import DebugIt
from PyTango.server import command
from .base import TangoRemoteProcessing


class TangoBenchmarker(TangoRemoteProcessing):
    """
    Device server for elmo_main.py-controller based
    """

    async def init_device(self):
        await super().init_device()
        self._durations = {}

    @DebugIt()
    @command(dtype_in=str, dtype_out=float)
    def get_duration(self, acquisition_name):
        return self._durations[acquisition_name]

    @DebugIt()
    @command(dtype_in=str)
    async def start_timer(self, acq_name):
        await self._process_stream(self.consume(acq_name))

    @DebugIt()
    @command()
    async def reset(self):
        self._durations = {}

    async def consume(self, acq_name):
        total_bytes = 0
        with PerformanceTracker() as pt:
            async for image in self._receiver.subscribe():
                total_bytes += image.nbytes
            pt.size = total_bytes * q.B
        self._durations[acq_name] = pt.duration.to(q.s).magnitude
