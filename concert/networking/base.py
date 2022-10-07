"""Connection protocols for network communication."""
import asyncio
import os
import logging
import numpy as np
import time
import zmq
import zmq.asyncio
from concert.quantities import q
from concert.config import AIODEBUG, PERFDEBUG


LOG = logging.getLogger(__name__)


class SocketConnection(object):

    """A two-way socket connection. *return_sequence* is a string appended
    after every command indicating the end of it, the default value
    is a newline (\\n).
    """

    def __init__(self, host, port, return_sequence="\n"):
        self._peer = (host, port)
        self._reader = None
        self._writer = None
        self.lock = asyncio.Lock()
        self.return_sequence = return_sequence

    async def __aenter__(self):
        LOG.log(AIODEBUG, 'Socket connection enter')
        await self.connect()

        return self

    async def __aexit__(self, exc_type, exc, tb):
        LOG.log(AIODEBUG, 'Socket connection exit')
        await self.close()

    async def connect(self):
        """Open connection."""
        self._reader, self._writer = await asyncio.open_connection(*self._peer)

    async def close(self):
        """Close connection."""
        self._writer.close()
        await self._writer.wait_closed()

    async def send(self, data):
        """
        Send *data* to the peer. The return sequence characters
        are appended to the data before it is sent.
        """
        if not self._reader:
            await self.connect()

        LOG.debug('Sending {0}'.format(data))
        data += self.return_sequence
        self._writer.write(data.encode('ascii'))
        await self._writer.drain()

    async def recv(self, num=1024):
        """
        Read *num* bytes from the socket. The result is first stripped from the trailing return
        sequence characters and then returned.
        """
        if not self._reader:
            await self.connect()

        result = (await self._reader.read(num)).decode('ascii')
        if result.endswith(self.return_sequence):
            # Strip the command-ending character
            result = result.rstrip(self.return_sequence)
        LOG.debug('Received {0}'.format(result))
        return result

    async def execute(self, data, num=1024):
        """Execute command and wait for response (coroutine-safe, not thread-safe). Read *num* bytes
        from the socket.
        """
        async with self.lock:
            await self.send(data)
            result = await self.recv(num=num)

        return result


def get_tango_device(uri, peer=None, timeout=10 * q.s):
    """
    Get a Tango device by specifying its *uri*. If *peer* is given change the tango_host specifying
    which database to connect to. Format is host:port as a string. *timeout* sets the device's
    general timeout. It is converted to milliseconds, converted to integer and then the tango
    device's `set_timout_millis` is called with the converted integer value.
    """
    import IPython
    import PyTango

    if peer is not None:
        os.environ["TANGO_HOST"] = peer

    executor = None
    if IPython.version_info >= (8, 0):
        from IPython.core.async_helpers import get_asyncio_loop
        ipython_loop = get_asyncio_loop()
        executor = PyTango.asyncio_executor.AsyncioExecutor(loop=ipython_loop)

    device = PyTango.DeviceProxy(
        uri, green_mode=PyTango.GreenMode.Asyncio, asyncio_executor=executor
    )
    device.set_timeout_millis(int(timeout.to(q.ms).magnitude))

    return device


def zmq_create_image_metadata(image):
    """Create metadata needed for sending *image* over zmq."""
    if image is None:
        return {'end': True}
    else:
        return {
            'dtype': str(image.dtype),
            'shape': image.shape,
        }


async def zmq_receive_image(socket, return_metadata=False):
    """Receive image data from a zmq *socket*."""
    metadata = await socket.recv_json()
    if 'end' in metadata:
        return (metadata, None) if return_metadata else None

    msg = await socket.recv(copy=False)
    array = np.frombuffer(msg, dtype=metadata['dtype']).reshape(metadata['shape'])

    return (metadata, array) if return_metadata else array


async def zmq_send_image(socket, image, metadata=None):
    """Send *image* over zmq *socket*. If *metadata* is None, it is created. If *image* is None,
    {'end': True} is sent in metadata and no actual payload is sent.
    """
    if image is None or not metadata:
        metadata = zmq_create_image_metadata(image)

    try:
        if image is None:
            await socket.send_json(metadata, zmq.NOBLOCK)
        else:
            await socket.send_json(metadata, zmq.NOBLOCK | zmq.SNDMORE)
            await socket.send(image, flags=zmq.NOBLOCK, copy=False)
    except zmq.Again:
        LOG.debug('No listeners or queue full on %s', socket.get(zmq.LAST_ENDPOINT))


class ZmqBase:

    """
    Base for sending/receiving zmq image streams.

    :param endpoint: endpoint in form transport://address
    """

    def __init__(self, endpoint=None):
        self._endpoint = endpoint
        self._context = zmq.asyncio.Context()
        self._socket = None
        if endpoint:
            self.connect(endpoint)

    @property
    def endpoint(self):
        """endpoint in form transport://address"""
        return self._endpoint

    def connect(self, endpoint):
        """Connect to an *endpoint* which must not be None. If it's the same one as the one we are
        connected to now nothing happens, otherwise current socket is disconnected and the new
        connection is made.
        """
        if not endpoint:
            raise RuntimeError('Cannot connect when endpoint is not specified')

        if self._socket:
            if endpoint == self._endpoint:
                # We are connected to the desired endpoint already
                return
            # New endpoint specified, first disconnect
            self.close()

        self._endpoint = endpoint
        self._setup_socket()

    def close(self):
        """Close the socket."""
        if self._socket:
            self._socket.close()
            self._socket = None

    def _setup_socket(self):
        """Create and connect zmq socket, implementation-specific."""
        raise NotImplementedError


class ZmqSender(ZmqBase):

    """
    Sender of zmq image streams.
    :param endpoint: endpoint in form transport://address
    :param sndhwm: high send water mark
    """

    def __init__(self, endpoint=None, sndhwm=None):
        self._sndhwm = sndhwm
        super().__init__(endpoint=endpoint)

    def _setup_socket(self):
        """Create and connect a PUSH socket."""
        self._socket = self._context.socket(zmq.PUSH)
        # Do not keep old images
        self._socket.setsockopt(zmq.LINGER, 0)
        if self._sndhwm:
            # TODO: use this if we cannot send and hang on blocking behavior or start getting Again
            # errors in case of NOBLOCK
            self._socket.set(zmq.SNDHWM, self._sndhwm)
        self._socket.bind(self._endpoint)

    async def send_image(self, image):
        """Send *image*."""
        await zmq_send_image(self._socket, image)


class ZmqReceiver(ZmqBase):

    """
    Receiver of zmq image streams.
    :param endpoint: endpoint in form transport://address
    :param timeout: wait this many milliseconds between checking for the finished state and trying
    to get the next image
    :param topic: topic filter for image subscription, works only in combination with
    *reliable=False*
    :param polling_timeout: wait this many milliseconds between asking for images
    """

    def __init__(self, endpoint=None, reliable=True, rcvhwm=1, topic='', polling_timeout=100):
        self._reliable = reliable
        self._rcvhwm = rcvhwm
        self._topic = topic
        self._poller = zmq.asyncio.Poller()
        self._polling_timeout = polling_timeout
        self._request_stop = False
        super().__init__(endpoint=endpoint)

    def _setup_socket(self):
        """Create and connect a PULL socket."""
        self._socket = self._context.socket(zmq.PULL if self._reliable else zmq.SUB)
        if not self._reliable:
            self._socket.set(zmq.RCVHWM, self._rcvhwm)
            self._socket.setsockopt_string(zmq.SUBSCRIBE, self._topic)
        self._socket.connect(self._endpoint)
        self._poller.register(self._socket, zmq.POLLIN)

    def stop(self):
        """Stop receiving data."""
        self._request_stop = True

    def close(self):
        if self._socket:
            self._poller.unregister(self._socket)
        super().close()

    async def is_message_available(self, polling_timeout=None):
        """Wait on the socket *polling_timeout* milliseconds and if an image is available return
        True, False otherwise. If *polling_timeout* is None, use the constructor value; -1 means
        infinity.
        """
        polling_timeout = self._polling_timeout if polling_timeout is None else polling_timeout
        sockets = dict(await self._poller.poll(timeout=polling_timeout))

        return self._socket in sockets and sockets[self._socket] == zmq.POLLIN

    async def receive_image(self, return_metadata=False):
        """Receive image."""
        return await zmq_receive_image(self._socket, return_metadata=return_metadata)

    async def subscribe(self, return_metadata=False):
        """Receive images."""
        i = 0
        finished = False

        try:
            while True:
                num_tries = 0
                while True:
                    if await self.is_message_available():
                        # There is something to consume
                        image = await self.receive_image(return_metadata=return_metadata)
                        if return_metadata:
                            metadata, image = image
                        break
                    elif self._request_stop:
                        break
                    else:
                        num_tries += 1
                if self._request_stop or image is None:
                    # Poison pill or stop requested
                    LOG.debug(
                        '%s stopping at i=%d (reason: %s, current tries=%d)',
                        self._endpoint,
                        i,
                        'stop requested' if self._request_stop else 'image=None',
                        num_tries
                    )
                    self._request_stop = False
                    break
                else:
                    LOG.log(
                        PERFDEBUG,
                        'i=%d (current tries=%d [%.3f ms])',
                        i + 1, num_tries, self._polling_timeout * num_tries * 1e-3
                    )
                    yield (metadata, image) if return_metadata else image
                i += 1
        except BaseException as e:
            LOG.debug(
                '%s stopping at i=%d (reason: %s, current tries=%d)',
                self._endpoint,
                i,
                e.__class__.__name__,
                num_tries
            )
            raise
        finally:
            # Make sure no images are lingering around
            flush_i = 0
            while await self.is_message_available():
                await self._socket.recv()
                flush_i += 1
            if flush_i:
                LOG.debug("Flushed %d messages", flush_i)


class BroadcastServer(ZmqReceiver):
    """
    A ZMQ server which listens to some remote host and broadcasts the data further to the specified
    endpoints. If *reliable* is True, use PUSH/PULL, otherwise PUB/SUB (in which case
    *broadcast_endpoints* must be a list of just one endpoint).

    :param endpoint: data source endpoint in form transport://address
    :param broadcast_endpoints: tuples of destination endpoints in form (transport://address,
    reliable, sndhwm), if *reliable* is True use PUSH, otherwise PUB socket type; *sndhwm* is the
    high water mark (use 1 for always getting the newest image, only applicable for non-reliable
    case)
    """
    def __init__(self, endpoint, broadcast_endpoints, polling_timeout=100):
        super().__init__(endpoint, polling_timeout=polling_timeout)
        self._broadcast_sockets = set([])
        self._poller_out = zmq.asyncio.Poller()
        self._finished = None
        self._request_stop_forwarding = False

        for (destination, reliable, sndhwm) in broadcast_endpoints:
            socket = self._context.socket(zmq.PUSH if reliable else zmq.PUB)
            if not reliable:
                socket.set(zmq.SNDHWM, sndhwm)
            socket.bind(destination)
            self._broadcast_sockets.add(socket)
            self._poller_out.register(socket, flags=zmq.POLLOUT)

    async def _forward_image(self, image, metadata):
        # Until zmq 23.2.1, this would take the whole *timeout* time even if there were events on
        # the sockets
        sockets = dict(await self._poller_out.poll(timeout=self._polling_timeout))
        if sockets.keys() != self._broadcast_sockets:
            dead_ends = [
                socket.get(zmq.LAST_ENDPOINT) for socket in self._broadcast_sockets
                if socket not in sockets
            ]
            LOG.warning(
                "Cannot forward to: %s",
                ",".join([endpoint.decode('ascii') for endpoint in dead_ends])
            )
        await asyncio.gather(
            *(zmq_send_image(
                socket,
                image,
                metadata=metadata
            ) for socket in self._broadcast_sockets)
        )

    async def consume(self):
        """Receive data from server and broadcast to all consumers."""
        if self._request_stop_forwarding:
            raise BroadcastError('Cannot consume streams with shutdown BroadcastServer')

        LOG.debug('BroadcastServer: forwarding new stream')
        self._finished = asyncio.Event()
        st = None
        try:
            i = 1
            async for (metadata, image) in self.subscribe(return_metadata=True):
                if st is None:
                    st = time.perf_counter()
                if self._request_stop_forwarding:
                    break
                await self._forward_image(image, metadata)
                LOG.log(PERFDEBUG, "Forwarded image %d", i)
                i += 1

            # End of stream
            await self._forward_image(None, None)
        finally:
            self._finished.set()
            if st is None:
                st = time.perf_counter()
            LOG.debug('BroadcastServer: stream finished in %.3f s', time.perf_counter() - st)

    async def serve(self):
        """Serve until asked to stop."""
        while True:
            # Wait until actual data is available in case someone requests stop between one stream
            # end and second stream start
            await self.is_message_available(polling_timeout=-1)
            if self._request_stop_forwarding:
                break
            await self.consume()

    async def shutdown(self):
        """Shutdown forwarding forever. It is not possible to call *consume()* from now on. If this
        has been called before it has no effect.
        """
        if self._request_stop_forwarding:
            # Already shutdown
            return
        self._request_stop_forwarding = True

        if self._finished and not self._finished.is_set():
            # We have been started and are not finished
            self.stop()
            # Wait for the forwarding to finish gracefully
            await self._finished.wait()

        for socket in self._broadcast_sockets:
            socket.close()

        LOG.info('BroadcastServer: shut down')


def is_zmq_endpoint_local(endpoint):
    """Return True if *endpoint* belongs to this machine."""
    prefixes = ['tcp://localhost', 'tcp://127.0.0.1', 'ipc://']

    for prefix in prefixes:
        if endpoint.startswith(prefix):
            return True

    return False


class BroadcastError(Exception):
    """BroadcastServer-related exceptions."""
    pass
