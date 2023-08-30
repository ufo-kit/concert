"""
A :class:`Camera` can be set via the device-specific properties that can be set
and read with :meth:`.Parameter.set` and :meth:`.Parameter.get`.  Moreover, a
camera provides means to

* :meth:`~Camera.start_recording` frames,
* :meth:`~Camera.stop_recording` the acquisition,
* :meth:`~Camera.trigger` a frame capture and
* :meth:`~Camera.grab` to get the last frame.

Camera triggering is specified by the :attr:`~Camera.trigger_source` parameter, which
can be one of

* :attr:`camera.trigger_sources.AUTO` means the camera triggers itself
  automatically, the frames start being recorded right after the
  :meth:`~camera.start_recording` call and stop being recorded by
  :meth:`~camera.stop_recording`

* :attr:`Camera.trigger_sources.SOFTWARE` means the camera needs to be triggered
  by the user by :meth:`~Camera.trigger()`. This way you have complete programatic
  control over when is the camera triggered, example usage::

    camera.trigger_source = camera.trigger_sources.SOFTWARE
    start_recording(camera)
    trigger(camera)
    long_operation()
    # Here we get the frame from before the long operation
    grab(camera)

* :attr:`Camera.trigger_sources.EXTERNAL` is a source when the camera is triggered
  by an external low-level signal (such as TTL). This source provides very precise
  triggering in terms of time synchronization with other devices


To setup and use a camera in a typical environment, you would do::

    import numpy as np
    from concert.devices.cameras.uca import Camera

    camera = Camera('pco')
    camera.trigger_source = camera.trigger_sources.SOFTWARE
    camera.exposure_time = 0.2 * q.s
    start_recording(camera)
    trigger(camera)
    data = grab(camera)
    stop_recording(camera)

    print("mean=%f, stddev=%f" % (np.mean(data), np.std(data)))

You can apply primitive operations to the frames obtained by :meth:`Camera.grab` by setting up a
:attr:`Camera.convert` attribute to some callable which takes just one argument which is the grabbed
frame. The callable is applied to the frame and the converted one is returned by
:meth:`Camera.grab`. You can do::

    import numpy as np
    from concert.devices.cameras.dummy import Camera

    camera = Camera()
    camera.convert = np.fliplr
    # The frame is left-right flipped
    grab(camera)
"""
import asyncio
import contextlib
import logging
from concert.base import AccessorNotImplementedError, Parameter, Quantity, State, check, identity
from concert.config import AIODEBUG
from concert.coroutines.base import background
from concert.quantities import q
from concert.helpers import Bunch
from concert.devices.base import Device


LOG = logging.getLogger(__name__)


class CameraError(Exception):

    """Camera specific errors."""
    pass


class Camera(Device):

    """Base class for remotely controllable cameras.

    .. py:attribute:: frame-rate

        Frame rate of acquisition in q.count per time unit.
    """

    trigger_sources = Bunch(['AUTO', 'SOFTWARE', 'EXTERNAL'])
    trigger_types = Bunch(['EDGE', 'LEVEL'])
    state = State(default='standby')
    frame_rate = Quantity(1 / q.second, help="Frame frequency")
    trigger_source = Parameter(help="Trigger source")

    async def __ainit__(self):
        await super(Camera, self).__ainit__()
        self.convert = identity
        self._grab_lock = asyncio.Lock()

    @background
    @check(source='standby', target='recording')
    async def start_recording(self):
        """
        start_recording()

        Start recording frames.
        """
        await self._record_real()

    @background
    @check(source='recording', target='standby')
    async def stop_recording(self):
        """
        stop_recording()

        Stop recording frames.
        """
        async with self._grab_lock:
            await self._stop_real()

    @contextlib.asynccontextmanager
    async def recording(self):
        """
        recording()

        A context manager for starting and stopping the camera.

        In general it is used with the ``async with`` keyword like this::

            async with camera.recording():
                frame = await camera.grab()
        """
        await self.start_recording()
        try:
            yield
        finally:
            LOG.log(AIODEBUG, 'stop recording in recording()')
            await self.stop_recording()

    @background
    @check(source='recording')
    async def trigger(self):
        """Trigger a frame if possible."""
        await self._trigger_real()

    @background
    @check(source=['recording', 'readout'])
    async def grab(self):
        """Return a NumPy array with data of the current frame."""
        async with self._grab_lock:
            # Make sure there are no two concurrent grabs (e.g. we are streaming in a separate task
            # and someone calls grab() in the session
            return self.convert(await self._grab_real())

    # Be strict, if the camera is recording an experiment might be in progress, so let's restrict
    # this to 'standby'
    @check(source=['standby'])
    async def stream(self):
        """
        stream()

        Grab frames continuously yield them. This is an async generator.
        """
        await self['trigger_source'].stash()
        await self.set_trigger_source(self.trigger_sources.AUTO)
        await self.start_recording()

        try:
            while True:
                # Make state checking and grabbing atomic so that no one can stop_recording()
                # between the state is obtained and grab() is called.
                async with self._grab_lock:
                    if await self.get_state() == 'recording':
                        image = self.convert(await self._grab_real())
                    else:
                        break
                yield image
        except asyncio.CancelledError:
            if await self.get_state() == 'recording':
                await self.stop_recording()
        finally:
            await self['trigger_source'].restore()

    async def _get_trigger_source(self):
        raise AccessorNotImplementedError

    async def _set_trigger_source(self, source):
        raise AccessorNotImplementedError

    async def _record_real(self):
        raise AccessorNotImplementedError

    async def _stop_real(self):
        raise AccessorNotImplementedError

    async def _trigger_real(self):
        raise AccessorNotImplementedError

    async def _grab_real(self):
        raise AccessorNotImplementedError


class BufferedMixin(Device):

    """A camera that stores the frames in an internal buffer"""

    state = State(default='standby')

    @background
    @check(source='standby', target='readout')
    async def start_readout(self):
        """
        start_readout()

        Start reading out frames.
        """
        await self._start_readout_real()

    @background
    @check(source='readout', target='standby')
    async def stop_readout(self):
        """
        stop_readout()

        Stop reading out frames.
        """
        await self._stop_readout_real()

    @contextlib.asynccontextmanager
    async def readout(self):
        """
        readout()

        A context manager for starting and stopping the readout.

        In general it is used with the ``async with`` keyword like this::

            async with camera.readout():
                frames = await camera.readout_buffer()
        """
        await self.start_readout()
        try:
            yield
        finally:
            LOG.log(AIODEBUG, 'stop readout in readout()')
            await self.stop_readout()

    @check(source='readout')
    async def readout_buffer(self, *args, **kwargs):
        async for item in self._readout_real(*args, **kwargs):
            yield item

    async def _start_readout_real(self):
        raise AccessorNotImplementedError

    async def _stop_readout_real(self):
        raise AccessorNotImplementedError

    async def _readout_real(self, *args, **kwargs):
        raise AccessorNotImplementedError


class RemoteMixin:

    """
    A remote camera which can grab more frames at once and instead of returning them to concert they
    are processed otherwise, e.g. sent over network to some consumer.
    """

    remote = True

    @background
    async def grab_many(self, num):
        async with self._grab_lock:
            try:
                await self._grab_many_real(num)
            except asyncio.CancelledError:
                await self._stop_streaming()
                raise

    @background
    @check(source=['recording', 'readout'])
    async def grab(self):
        """Grab a frame remotely, no conversion happens as opposed to local cameras."""
        async with self._grab_lock:
            await self._grab_real()

    async def _grab_real(self):
        await self._grab_many_real(1)

    async def _grab_many_real(self, num):
        raise NotImplementedError

    async def _stop_streaming(self):
        """
        Stop sending images. The server must send a poison pill which serves as an end-of-stream
        indicator to consumers.
        """
        raise NotImplementedError
