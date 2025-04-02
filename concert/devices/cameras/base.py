"""
A :class:`Camera` can be set via the device-specific properties that can be set
and read with :meth:`.Parameter.set` and :meth:`.Parameter.get`.  Moreover, a
camera provides means to

* :meth:`~Camera.start_recording` frames,
* :meth:`~Camera.stop_recording` the acquisition,
* :meth:`~Camera.trigger` a frame capture and
* :meth:`~Camera.grab` to get the last frame.
* :meth:`~Camera.register_endpoint` to register a ZMQ server endpoint
* :meth:`~Camera.unregister_endpoint` to remove a previously registered ZMQ server endpoint

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

The image can be mirrored by setting the parameter *mirror* to true.
The Image can be rotated (0-3 times) be setting the parameter *rotate* to the number of rotations.
In case of a remote camera, the raw frames and these parameters in the metadata are transmitted.

Cameras can send images over a ZMQ stream by the :meth:`Camera.grab_send` method. Instead of giving
the frames to the user, it sends the frames via the ZMQ stream. For this to work once must do the
ZMQ server endpoints registration with the camera using `register_endpoint` method, which takes
an instance of `concert.helpers.CommData` as argument.

There is a grab :class:`asyncio.Lock` which prevents the frames to be grabbed at the same time from
competing methods like :meth:`Camera.grab` and :meth:`RemoteMixin.grab_send`.
"""
import asyncio
import contextlib
import logging
from abc import abstractmethod

from typing import Dict
import zmq

from concert.base import AccessorNotImplementedError, Parameter, Quantity, State, check
from concert.config import AIODEBUG
from concert.coroutines.base import background
from concert.quantities import q
from concert.helpers import Bunch, CommData, ImageWithMetadata
from concert.devices.base import Device
from concert.networking.base import ZmqSender


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
    _senders: Dict[CommData, ZmqSender]
    mirror = Parameter(help="Mirror the image")
    rotate = Parameter(help="Rotate the image")

    async def __ainit__(self):
        self._rotate = 0
        self._mirror = False
        await super(Camera, self).__ainit__()
        self._grab_lock = asyncio.Lock()
        self._senders = {}

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

        Stop recording frames, acquires the grab lock before the actual implementation is called.
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
    async def grab(self) -> ImageWithMetadata:
        """Return a concert.storage.ImageWithMetadata (subclass of np.ndarray) with data of the
        current frame. Acquires grab lock."""
        async with self._grab_lock:
            img = await self._grab_real()
            meta = {"mirror": await self.get_mirror(), "rotate": await self.get_rotate()}
            return ImageWithMetadata(img, metadata=meta).convert()

    # Be strict, if the camera is recording an experiment might be in progress, so let's restrict
    # this to 'standby'
    @check(source=['standby'])
    async def stream(self):
        """
        stream()

        Grab frames continuously yield them. This is an async generator. Acquires grab lock in every
        iteration separately, i.e. you can e.g. call :meth:`.stop_recording` while :meth:`.stream`
        runs in the background.
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
                        image = await self._grab_real()
                        meta = {"mirror": await self.get_mirror(), "rotate": await self.get_rotate()}
                        image = ImageWithMetadata(image, metadata=meta).convert()
                    else:
                        break
                yield image
        except asyncio.CancelledError:
            if await self.get_state() == 'recording':
                await self.stop_recording()
        finally:
            await self['trigger_source'].restore()

    @background
    @check(source=['recording', 'readout'])
    async def grab_send(self, num, end=True):
        """Grab and send over a zmq socket. If *end* is True, end-of-stream indicator is sent to all
        consumers when the desired number of images is sent. Acquires grab lock for the whole time
        *num* frames are being sent.
        """
        async with self._grab_lock:
            try:
                await self._grab_send_real(num, end=end)
            except asyncio.CancelledError:
                await self.stop_sending()
                raise

    @background
    @check(source='recording')
    async def stop_sending(self):
        """
        Stop sending images. The server must send a poison pill which serves as an end-of-stream
        indicator to consumers.
        """
        await self._stop_sending()

    async def _grab_send_real(self, num, end=True):
        async def send_to_all(image):
            await asyncio.gather(
                *(sender.send_image(image) for sender in self._senders.values())
            )

        for _ in range(num):
            img = await self._grab_real()
            meta = {"mirror": await self.get_mirror(), "rotate": await self.get_rotate()}
            img = ImageWithMetadata(img, metadata=meta)
            await send_to_all(img)

        if end:
            await send_to_all(None)

    async def _stop_sending(self):
        raise AccessorNotImplementedError

    async def unregister_endpoint(self, endpoint: CommData) -> None:
        """
        Removes a previously registered ZMQ server endpoint.

        :param endpoint: previously registered ZMQ server endpoint
        :type endpoint: concert.helpers.CommData
        """
        if endpoint in self._senders:
            self._senders[endpoint].close()
            del self._senders[endpoint]

    async def register_endpoint(self, endpoint: CommData) -> None:
        """
        Registers a ZMQ server endpoint to stream captured frames to a remote client consumer.

        :param endpoint: ZMQ server endpoint to register
        :type endpoint: concert.helpers.CommData
        """
        if endpoint in self._senders:
            raise ValueError("zmq endpoint already in list")

        self._senders[endpoint] = ZmqSender(
            endpoint.server_endpoint,
            reliable=endpoint.socket_type == zmq.PUSH,
            sndhwm=endpoint.sndhwm
        )

    async def unregister_all(self) -> None:
        for sender in self._senders.values():
            sender.close()
        self._senders = {}

    @abstractmethod
    async def _get_trigger_source(self):
        ...

    @abstractmethod
    async def _set_trigger_source(self, source):
        ...

    @abstractmethod
    async def _record_real(self):
        ...

    @abstractmethod
    async def _stop_real(self):
        ...

    @abstractmethod
    async def _trigger_real(self):
        ...

    @abstractmethod
    async def _grab_real(self) -> ImageWithMetadata:
        ...

    async def _set_mirror(self, mirror):
        self._mirror = bool(mirror)

    async def _get_mirror(self):
        return self._mirror

    async def _set_rotate(self, r):
        self._rotate = int(r)

    async def _get_rotate(self):
        return self._rotate


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

    @abstractmethod
    async def _start_readout_real(self):
        ...

    @abstractmethod
    async def _stop_readout_real(self):
        ...

    @abstractmethod
    async def _readout_real(self, *args, **kwargs):
        ...
