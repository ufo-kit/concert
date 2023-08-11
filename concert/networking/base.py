"""Connection protocols for network communication."""
import asyncio
import os
import logging
from typing import Optional, List, Tuple, AsyncIterable, Set
import numpy as np
import time
import zmq
from numpy.typing import NDArray
import zmq.asyncio as zao
from pint import UnitRegistry
from concert.quantities import q
from concert.config import AIODEBUG, PERFDEBUG
from concert.networking.typing import Metadata_t, Payload_t, Subscription_t
from concert.networking.typing import ConcertDeviceProxy, BroadcastError

LOG = logging.getLogger(__name__)


class SocketConnection(object):
    """
    A two-way socket connection. *return_sequence* is a string appended
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


def get_tango_device(
        uri: str,
        peer: Optional[str] = None,
        timeout: UnitRegistry = 10 * q.s) -> ConcertDeviceProxy:
    """
    Get a Tango device.

    :param uri: device identifier for the tango device
    :type uri: str
    :param peer: if *peer* is given change the tango_host specifying which database to connect to.
    Format is host:port as a string
    :type peer: Optional[str]
    :param timeout: sets the device's general timeout. It is converted to milliseconds,
    (converted to integer and then the tango device's `set_timout_millis` is called with
    the converted integer value)
    :type timeout: UnitRegistry
    :return: an abstract tango device which lets users write arbitrary attribute as
    key value pairs.
    :rtype: ConcertDeviceProxy
    """
    import IPython
    import PyTango

    if peer is not None:
        # Specifies the location of the tango database server. Its default format is
        # "<host>:<port>"
        # https://tango-controls.readthedocs.io/en/latest/development/advanced/reference.html#tango-host
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


def zmq_create_image_metadata(image: Optional[NDArray]) -> Metadata_t:
    """
    Create metadata needed for sending *image* over zmq.

    :param image: image captured by camera device, could be None
    :type image: Optional[NDArray]
    """
    if image is None:
        return {"end": True}
    else:
        return {
            "dtype": str(image.dtype),
            "shape": str(image.shape),
        }


async def zmq_receive_image(socket: zao.Socket, return_metadata: bool = False) -> Payload_t:
    """
    Receive image data from a zmq *socket*.

    :param socket: zmq socket connection
    :type socket: zao.Socket
    :param return_metadata: whether communication metadata would be propagated to caller
    :type return_metadata: bool
    :return: received image and optional metadata
    :rtype: RecvPayload_t
    """
    # TODO: Clarify - recv_json is not defined as a coroutine. Why use
    #   await ?
    metadata: Metadata_t = socket.recv_json()
    if "end" in metadata:  # type: ignore
        return (metadata, None) if return_metadata else None
    msg = await socket.recv(copy=False)  # type: ignore
    array: NDArray = np.frombuffer(
        msg, dtype=metadata["dtype"]).reshape(metadata["shape"])  # type: ignore
    return (metadata, array) if return_metadata else array


async def zmq_send_image(
        socket: zao.Socket,
        image: Optional[NDArray],
        metadata: Optional[Metadata_t] = None) -> None:
    """
    Sends *image* over zmq *socket*. If *metadata* is None, it is created.
    If *image* is None, {'end': True} is sent in metadata and no actual payload is sent.

    :param socket: zmq socket connection
    :type socket: zao.Socket
    :param image: image captured by camera device, could be None in case end of
    transmission
    :type image: Optional[NDArray]
    :param metadata: metadata to specify type or shape of the image or end of
    transmitting images
    :type metadata: Optional[SockCommMetaData_t]
    """
    if metadata is None:
        metadata = zmq_create_image_metadata(image)
    try:
        # TODO: Clarify: following methods from zmq.asyncio.Socket class methods are not
        #  defined to be awaitable. Why we are invoking them with `await` ?
        if image is None:
            socket.send_json(metadata, zmq.NOBLOCK)
        else:
            socket.send_json(metadata, zmq.NOBLOCK | zmq.SNDMORE)
            await socket.send(image, flags=zmq.NOBLOCK, copy=False)
    except zmq.Again:
        LOG.debug('No listeners or queue full on %s', socket.get(zmq.LAST_ENDPOINT))


class ZmqBase:
    """
    Base for sending/receiving zmq image streams.

    :param endpoint: socket connection endpoint in form transport://address
    :type endpoint: str
    """

    _endpoint: Optional[str]
    _context: zao.Context
    _socket: Optional[zao.Socket]

    def __init__(self, endpoint: Optional[str] = None) -> None:
        self._endpoint = endpoint
        self._context = zao.Context()
        self._socket = None
        if endpoint:
            self.connect(endpoint)

    @property
    def endpoint(self) -> Optional[str]:
        """endpoint in form transport://address"""
        return self._endpoint

    def connect(self, endpoint: str) -> None:
        """
        Connect to an *endpoint* which must not be None. If it's the same one as the one we are
        connected to now nothing happens, otherwise current socket is disconnected and the new
        connection is made.

        :param endpoint: endpoint having format transport://address
        :type endpoint: str
        """
        if not endpoint:
            raise RuntimeError('Cannot connect when endpoint is not specified')
        if self._socket:
            if endpoint == self._endpoint:
                # If we are connected to the desired endpoint already no action
                # is required
                return
            # If a new endpoint is specified disconnect from the old endpoint
            # before setting up a new socket connection
            self.close()
        self._endpoint = endpoint
        self._setup_socket()

    def close(self) -> None:
        """
        Closes the socket connection and invalidates the socket reference."""
        if self._socket:
            self._socket.close()
            self._socket = None

    def _setup_socket(self) -> None:
        """
        Creates and connect zmq socket. Derived class should provide the
        implementation"""
        raise NotImplementedError


class ZmqSender(ZmqBase):
    """
    Sender of zmq image streams.
    :param endpoint: endpoint in form transport://address
    :param snd_hwm: high send watermark, depicts a hard limit on the max number of
    outstanding messages that should be queued in memory for the intended socket
    communication peer: http://api.zeromq.org/2-1:zmq-setsockopt#toc3
    :type snd_hwm: int
    """

    _snd_hwm: Optional[int]

    def __init__(self,
                 endpoint: Optional[str] = None,
                 snd_hwm: Optional[int] = None) -> None:
        self._snd_hwm = snd_hwm
        super().__init__(endpoint=endpoint)

    def _setup_socket(self) -> None:
        """Create and connect a PUSH-type socket"""
        # We create a push-type socker connection at the sender end, means if there is a one
        # receiver peer, it would result in a strongly coupled messaging pattern and if there are
        # multiple receiver peers the messages from sender would be served in round-robin(turn
        # -based) fashion.
        self._socket = self._context.socket(zmq.PUSH)
        # Do not keep old images. Setting the ZMQ_LINGER option means
        # upon closing the socket connection all pending messages would
        # be discarded: http://api.zeromq.org/2-1:zmq-setsockopt#toc15
        self._socket.setsockopt(option=zmq.LINGER, value=0)
        if self._snd_hwm:
            # TODO: use this if we cannot send and hang on blocking behavior or start getting Again
            # errors in case of NOBLOCK
            # TODO: Clarify what its implications are. By default it is supposed to set a hard
            #   limit on the number of messages to keep in memory for intended recipient.
            self._socket.set(option=zmq.SNDHWM, value=self._snd_hwm)
        assert (self._endpoint is not None)
        self._socket.bind(addr=self._endpoint)

    async def send_image(self, image: Optional[NDArray]) -> None:
        """
        Send *image*.

        :param image: image captured by camera device, modelled by NumPy array, could
        be None if end of transmission
        :type image: NDArray
        """
        assert (self._socket is not None)
        await zmq_send_image(self._socket, image)


class ZmqReceiver(ZmqBase):
    """
    Receiver of zmq image streams.
    :param endpoint: endpoint in form transport://address
    :type endpoint: str
    :param reliable: determines whether receiver would act as a pull type receiver or as
    subscriber to a topic in accordance with decoupled pub-sub communication pattern
    :type reliable: bool
    :param rcv_hwm:
    :type rcv_hwm: int
    :param topic: topic filter for image subscription, works only in combination with
    :type topic: str
    :param polling_timeout: wait this many milliseconds between asking for images
    :type polling_timeout: int

    # TODO: What is timeout parameter, how it is used if at all ?
    :param timeout: wait this many milliseconds between checking for the finished state and trying
    to get the next image
    """

    _rcv_hwm: int
    _reliable: bool
    _topic: str
    _poller: zao.Poller
    _polling_timeout: int
    _request_stop: bool

    def __init__(self,
                 endpoint: Optional[str] = None,
                 reliable: bool = True,
                 rcv_hwm: int = 1,
                 topic: str = "",
                 polling_timeout: int = 100) -> None:
        self._reliable = reliable
        self._rcv_hwm = rcv_hwm
        self._topic = topic
        self._poller = zao.Poller()
        self._polling_timeout = polling_timeout
        self._request_stop = False
        super().__init__(endpoint=endpoint)

    def _setup_socket(self) -> None:
        """Creates and connects the receiver end socket connection"""
        # We create either a pull-type (if reliable parameter is set to true) or a subscriber-type
        # socket peer (if reliable parameter is set to false). In case of a pull-type socket we'd
        # end up in a more strongly coupled messaging pattern and in alternative case we can
        # expect sender to be broadcasting the message with a topic on which receiver is supposed
        # to subscribe and establish an asynchronous loosely-coupled communication pattern.
        self._socket = self._context.socket(zmq.PULL if self._reliable else zmq.SUB)
        if not self._reliable:
            # When we don't want a reliable receiver we set a hard limit on the number of messages
            # in the receiver buffer which are yet to be read.
            # TODO: How it should work in our context ?
            self._socket.set(zmq.RCVHWM, self._rcv_hwm)
            self._socket.setsockopt_string(zmq.SUBSCRIBE, self._topic)
        # Connect to the socker endpoint and start polling for incoming messages on the socket
        # endpoint
        assert (self._endpoint is not None)
        self._socket.connect(self._endpoint)
        self._poller.register(self._socket, zmq.POLLIN)

    def stop(self) -> None:
        """
        Stops receiving data.
        """
        # This flag is read inside the asynchronous subscribe function
        self._request_stop = True

    def close(self) -> None:
        """
        Stops polling for incoming data and triggers socket connection closure.
        """
        if self._socket:
            self._poller.unregister(self._socket)
        super().close()

    async def is_message_available(self, polling_timeout: Optional[int] = None) -> bool:
        """
        Wait on the socket *polling_timeout* milliseconds and if an image is available return
        True, False otherwise. If *polling_timeout* is None, use the constructor value; -1 means
        infinity i.e., poll indefinitely until a message is available.

        :param polling_timeout: timeout at the receiver end for waiting on the incoming message
        :type polling_timeout: Optional[int]
        :return: if new message is available
        :rtype: bool
        """
        polling_timeout = self._polling_timeout if polling_timeout is None else polling_timeout
        sockets = dict(await self._poller.poll(timeout=polling_timeout))
        # ZMQ_POLLIN is a zmq socket event whose presence tells us that at least one message
        # may be received from the socket without blocking: http://api.zeromq.org/2-1:zmq-poll#toc2
        return self._socket in sockets and sockets[self._socket] == zmq.POLLIN

    async def receive_image(self, return_metadata: bool = False) -> Payload_t:
        """
        Receive image.

        :param return_metadata: whether communication metadata would be propagated to caller
        :type return_metadata: bool
        :return: received image and optional metadata
        :rtype: RecvPayload_t
        """
        assert (self._socket is not None)
        return await zmq_receive_image(self._socket, return_metadata=return_metadata)

    async def subscribe(self, return_metadata: bool = False) -> AsyncIterable[Subscription_t]:
        """
        Receive images via subscription.

        :param return_metadata: whether communication metadata would be propagated to caller
        :type return_metadata: bool
        :return: asynchronous iterable collection of optional metadata and images
        :rtype: AsyncIterable
        """
        i = 0
        # finished = False  # TODO: Verify its purpose
        try:
            metadata: Metadata_t = {}
            image: Optional[NDArray] = None
            while True:
                num_tries = 0
                while True:
                    if await self.is_message_available():
                        # There is something to consume
                        received = await self.receive_image(return_metadata=return_metadata)
                        if return_metadata:
                            # If there is something to consume we expect the same to be a
                            # non-null iterable collection of associated metadata and the image
                            # so that we could unpack the same
                            assert (received is not None)
                            metadata, image = received
                        break
                    # An explicit stop invocation from external caller could lead to this
                    # case
                    elif self._request_stop:
                        break
                    # If we have neither received an explicit stop signal nor a new message is
                    # available at the socket endpoint we keep retrying
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
                    assert (image is not None)  # TODO: Verify if this is reasonable assumption
                    yield (metadata, image) if return_metadata else image
                i += 1
        except BaseException as e:
            LOG.debug(
                '%s stopping at i=%d (reason: %s, current tries=%d)',
                self._endpoint,
                i,
                e.__class__.__name__,
                num_tries  # noqa
            )
            raise
        finally:
            # Make sure no images are lingering around
            flush_i = 0
            while await self.is_message_available():
                assert (self._socket is not None)
                await self._socket.recv()
                flush_i += 1
            if flush_i:
                LOG.debug("Flushed %d messages", flush_i)


class BroadcastServer(ZmqReceiver):
    """
    A ZMQ server which listens to some remote host and broadcasts the data further to the
    specified endpoints. If *reliable* is True, use PUSH/PULL, otherwise PUB/SUB (in which case
    *broadcast_endpoints* must be a list of just one endpoint).

    :param endpoint: data source endpoint in form transport://address
    :type endpoint: str
    :param broadcast_endpoints: tuples of destination endpoints in the form (transport://address,
    reliable, snd_hwm), if *reliable* is True use PUSH, otherwise PUB socket type; *snd_hwm* is the
    high watermark (use 1 for always getting the newest image, only applicable for non-reliable
    case)
    :type broadcast_endpoints: List[Tuple[str, bool, int]]
    :param polling_timeout: wait this many milliseconds between asking for images
    :type polling_timeout: int
    """

    _broadcast_sockets: Set[zao.Socket]
    _poller_out: zao.Poller
    _finished: Optional[asyncio.Event]
    _request_stop_forwarding: bool

    def __init__(self,
                 endpoint: str,
                 broadcast_endpoints: List[Tuple[str, bool, int]],
                 polling_timeout: int = 100) -> None:
        super().__init__(endpoint, polling_timeout=polling_timeout)
        self._broadcast_sockets = set([])
        self._poller_out = zao.Poller()
        self._finished = None
        self._request_stop_forwarding = False
        # Iterate over each endpoint and create push or publisher type socket connection
        for (destination, reliable, snd_hwm) in broadcast_endpoints:
            socket: zao.Socket = self._context.socket(zmq.PUSH if reliable else zmq.PUB)
            if not reliable:
                socket.set(option=zmq.SNDHWM, value=snd_hwm)
            socket.bind(destination)
            self._broadcast_sockets.add(socket)
            self._poller_out.register(socket, flags=zmq.POLLOUT)

    async def _forward_image(self,
                             image: Optional[NDArray],
                             metadata: Optional[Metadata_t]) -> None:
        """
        Asynchronously send optional image and metadata to each destination socket endpoint
        which were registered in init. Image and metadata could be None."""
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

    async def consume(self) -> None:
        """
        Receive data from server and broadcast to all consumers.
        """
        if self._request_stop_forwarding:
            raise BroadcastError('Cannot consume streams with shutdown BroadcastServer')

        LOG.debug('BroadcastServer: forwarding new stream')
        self._finished = asyncio.Event()
        st: float = 0.
        try:
            i = 1
            async for (metadata, image) in self.subscribe(return_metadata=True):
                if st == 0.:
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
            # If st is still 0. at this point, it means we did not broadcast any image
            if st == 0.:
                st = time.perf_counter()
            LOG.debug(
                'BroadcastServer: stream finished in %.3f s', time.perf_counter() - st)

    async def serve(self) -> None:
        """
        Serve until asked to stop.
        """
        while True:
            # Wait until actual data is available in case someone requests stop between one stream
            # end and second stream start
            await self.is_message_available(polling_timeout=-1)
            if self._request_stop_forwarding:
                break
            await self.consume()

    async def shutdown(self) -> None:
        """
        Shutdown forwarding forever. It is not possible to call *consume()* from now on. If this
        has been called before it has no effect.
        """
        if self._request_stop_forwarding:
            # Already shutdown
            return
        self._request_stop_forwarding = True
        if self._finished and not self._finished.is_set():
            # If finished event is instantiated and its underlying flag indicates False (event
            # criterion is not met) i.e., broadcast started but not finished yet.
            self.stop()
            # Wait for the forwarding to finish gracefully. We wait until the consume coroutine
            # eventually calls set on the Event at the end of broadcast.
            await self._finished.wait()

        for socket in self._broadcast_sockets:
            socket.close()
        LOG.info('BroadcastServer: shut down')


def is_zmq_endpoint_local(endpoint):
    """
    Return True if *endpoint* belongs to this machine.
    """
    prefixes = ['tcp://localhost', 'tcp://127.0.0.1', 'ipc://']

    for prefix in prefixes:
        if endpoint.startswith(prefix):
            return True

    return False


if __name__ == "__main__":
    pass

