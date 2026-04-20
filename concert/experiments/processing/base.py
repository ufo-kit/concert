"""
A processing node, which encapsulates camera and functions as an abstract data source for
experiments. Frames acquired from camera is distributed into one or more named streams.
"""
import asyncio
import json
import time
from typing import AsyncGenerator, Dict, List, Optional
import numpy as np
import zmq
import zmq.asyncio as zio
from concert.devices.cameras.base import Camera
from concert.networking.base import ZmqSender, ZmqReceiver
from concert.helpers import ImageWithMetadata
from concert.typing import ArrayLike

# Define specifications for stream operations
# TODO Be explicit and use assertions
OpsSpec = Dict[str, object]
StreamDesc = Dict[str, object]


class GeneralFrameProducer:
    """
    ZMQ-backed general frame producer that encapsulates a camera object(or ZMQ endpoint to one), and
    exposes a control REP RPC for stream management and grab_send, publishes processed frames via a
    `ZmqSender`, and provides `producer_corofunc(name)` adapter for local consumers.
    """

    _camera: Camera | str
    _ctrl_bind_addr: str
    _pub_bind_addr: str
    _streams: Dict[str, StreamDesc]
    _running: bool
    _ctx: Optional[zio.Context]
    _sock_rpc: Optional[zio.Socket]  # Socket for Remote Procedure Call
    _sender: Optional[ZmqSender]
    _receiver: Optional[ZmqReceiver]
    _ctrl_task: Optional[asyncio.Task]
    _pub_task: Optional[asyncio.Task]
    _frame_count: int
    _capture_lock: asyncio.Lock  # Serializes on-demand captures vs the background publisher


    def __init__(
            self,
            source: Camera | str,
            ctrl_bind_addr: str = "tcp://*:5555",
            pub_bind_addr: str = "tcp://*:5556") -> None:
        self._camera = source
        self._streams: Dict[str, StreamDesc] = {}
        self._ctrl_bind_addr = ctrl_bind_addr
        self._pub_bind_addr = pub_bind_addr
        self._running = False
        self._ctx = None
        self._sock_rpc = None     
        self._sender = None
        self._receiver = None
        self._ctrl_task = None
        self._pub_task = None
        self._frame_count = 0
        self._capture_lock = asyncio.Lock()

    # -------------------------
    # Lifecycle
    # -------------------------
    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._ctx = zio.Context.instance()
        assert self._ctx and isinstance(self._ctx, zio.Context)
        self._sock_rpc = self._ctx.socket(zmq.REP)
        assert self._sock_rpc and isinstance(self._ctx, zio.Socket)
        self._sock_rpc.bind(self._ctrl_bind_addr)
        # We create sender (socket.bind) end of the socket with zmq.PUB (unreliable) to facilitate
        # the broadcast pattern, where there can be several subscribers (socket.connect) receiving
        # the same frames.
        self._sender = ZmqSender(endpoint=self._pub_bind_addr, reliable=False, sndhwm=0)
        # If camera is a ZMQ endpoint string, we create receiver (socket.connect) with zmq.PULL
        # (reliable) to receive all frames from the camera (socket.bind) without frame drop. If
        # instead a `concert.devices.cameras.base.Camera` is provided then we use utility methods
        # from the camera instance directly.
        if isinstance(self._camera, str):
            self._receiver = ZmqReceiver(endpoint=self._camera, reliable=True, rcvhwm=0)
        else:
            self._receiver = None
        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        # Creating these tasks means that the _control_loop and _publish_loop coroutines are
        # scheduled on the event loop immediately. Current coroutine i.e., start(self) get's the
        # control back, hence can go on with its subsequent instructions (which in this case is to
        # finish).
        self._ctrl_task = loop.create_task(self._control_loop())
        self._pub_task = loop.create_task(self._publish_loop())

    async def stop(self):
        self._running = False
        # We await previously scheduled tasks on the event loop to say that now we are doing a
        # blocking wait for these scheduled coroutines to finish and then only stop(self) can go on
        # and execute the subsequent instructions.
        if self._ctrl_task:
            await self._ctrl_task
            self._ctrl_task = None
        if self._pub_task:
            await self._pub_task
            self._pub_task = None
        # Deal with the remote procedure call sockets.
        if self._sock_rpc:
            try:
                self._sock_rpc.close(linger=0)
            except Exception:
                pass
            self._sock_rpc = None
        # We close the sender ZMQ socket connection.
        if self._sender:
            try:
                self._sender.close()
            except Exception:
                pass
            self._sender = None
        # We close the receiver ZMQ socket connection.
        if self._receiver:
            try:
                self._receiver.close()
            except Exception:
                pass
            self._receiver = None
        self._ctx = None

    # -------------------------
    # control RPC
    # -------------------------
    async def _control_loop(self):
        """
        JSON-REQ/REP control:
        commands: add_stream, remove_stream, update_stream, list_streams,
                  get_pub_endpoint, grab_send
        """
        while self._running and self._sock_rpc:
            try:
                msg = await self._sock_rpc.recv()
            except zmq.error.ZMQError:
                break

            try:
                req = json.loads(msg.decode())
                cmd = req.get("cmd")
                if cmd == "add_stream":
                    name = req["name"]
                    pipeline = req.get("pipeline", [])
                    bp = req.get("backpressure", "drop_oldest")
                    if name in self._streams:
                        resp = {"ok": False, "error": "stream_exists"}
                    else:
                        self._streams[name] = {"pipeline": pipeline, "bp": bp}
                        resp = {"ok": True}
                elif cmd == "remove_stream":
                    name = req["name"]
                    self._streams.pop(name, None)
                    resp = {"ok": True}
                elif cmd == "update_stream":
                    name = req["name"]
                    pipeline = req.get("pipeline", [])
                    if name not in self._streams:
                        resp = {"ok": False, "error": "unknown_stream"}
                    else:
                        self._streams[name]["pipeline"] = pipeline
                        resp = {"ok": True}
                elif cmd == "list_streams":
                    resp = {"ok": True, "streams": {k: v["pipeline"] for k, v in self._streams.items()}}
                elif cmd == "get_pub_endpoint":
                    resp = {"ok": True, "pub_bind": self._pub_bind_addr}
                elif cmd == "grab_send":
                    # synchronous: run capture and reply after completion
                    number = int(req.get("number", 0))
                    stream = req.get("stream", "radios")
                    end = bool(req.get("end", True))
                    try:
                        frames = await self.grab_send(number, stream=stream, end=end)
                        resp = {"ok": True, "frames": frames}
                    except Exception as exc:
                        resp = {"ok": False, "error": str(exc)}
                else:
                    resp = {"ok": False, "error": "unknown_cmd"}
            except Exception as exc:
                resp = {"ok": False, "error": str(exc)}

            await self._sock_rpc.send(json.dumps(resp).encode())

    # -------------------------
    # background publish loop
    # -------------------------
    async def _publish_loop(self):
        """
        A background publisher that continuously consumes camera stream (if desired)
        and publishes processed outputs. If an on-demand capture is active (grab_send),
        this loop yields to avoid competing for frames.
        """
        cam_obj = self._camera

        # try to stop camera recording if camera object supports it
        if hasattr(cam_obj, "get_state") and awaitable(cam_obj.get_state):
            try:
                if await cam_obj.get_state() == "recording":
                    await cam_obj.stop_recording()
            except Exception:
                pass

        use_remote_cam = self._receiver is not None

        while self._running and self._sender:
            # don't compete when grab_send is running
            if self._capture_lock.locked():
                await asyncio.sleep(0.01)
                continue

            try:
                if use_remote_cam:
                    # consume remote camera stream continuously
                    async for in_meta, frame in self._receiver.subscribe(return_metadata=True):
                        if not self._running or self._capture_lock.locked():
                            break
                        await self._handle_frame_and_publish(frame)
                else:
                    # local camera object: pull frames if possible
                    if hasattr(cam_obj, "grab"):
                        frame = await cam_obj.grab()
                        await self._handle_frame_and_publish(frame)
                    else:
                        await asyncio.sleep(0.01)
            except Exception:
                await asyncio.sleep(0.005)

    async def _handle_frame_and_publish(self, frame):
        self._frame_count += 1
        ts = time.time()
        for name, meta in list(self._streams.items()):
            try:
                processed = await self._apply_pipeline(frame, meta.get("pipeline", []), self._frame_count)
                if processed is None:
                    continue
                arr = np.asarray(processed)
                meta_dict = {
                    "dtype": str(arr.dtype),
                    "shape": list(arr.shape),
                    "counter": self._frame_count,
                    "ts": ts,
                    "stream": name
                }
                await self._sender.send_image(ImageWithMetadata(arr, metadata=meta_dict))
            except Exception:
                continue

    # -------------------------
    # on-demand capture API (exposed to RemoteAutoDAQMixin)
    # -------------------------
    async def grab_send(self, number: int, stream: str = "radios", end: bool = True) -> int:
        """
        Capture `number` frames from the encapsulated camera and publish processed
        outputs to configured streams. This function blocks until `number` frames
        have been consumed and processed and then returns the number.
        """
        async with self._capture_lock:
            # ensure stream exists (optional convenience)
            if stream not in self._streams:
                self._streams[stream] = {"pipeline": [], "bp": "drop_oldest"}

            cam = self._camera
            count = 0
            use_remote_cam = self._receiver is not None

            # If camera is remote endpoint, use cam_receiver.subscribe and read N frames
            if use_remote_cam:
                async for in_meta, frame in self._receiver.subscribe(return_metadata=True):
                    await self._handle_frame_and_publish(frame)
                    count += 1
                    if count >= number:
                        break
            else:
                # local camera object: control trigger/recording
                if hasattr(cam, "set_trigger_source"):
                    await cam.set_trigger_source("AUTO")
                # use camera.recording() context if available
                if hasattr(cam, "recording"):
                    async with cam.recording():
                        for _ in range(number):
                            frame = None
                            if hasattr(cam, "grab"):
                                frame = await cam.grab()
                            else:
                                raise NotImplementedError("Camera object lacks grab()")
                            await self._handle_frame_and_publish(frame)
                            count += 1
                else:
                    # fallback: loop grabbing
                    for _ in range(number):
                        if hasattr(cam, "grab"):
                            frame = await cam.grab()
                            await self._handle_frame_and_publish(frame)
                            count += 1
                        else:
                            raise NotImplementedError("Camera cannot be driven by producer")

            # send end marker if requested
            if end and self._sender:
                await self._sender.send_image(None)

            return count

    # -------------------------
    # pipeline executor (minimal)
    # -------------------------
    async def _apply_pipeline(self, frame, pipeline: List[OpsSpec], counter: int):
        """
        Minimal declarative pipeline executor. Supported ops:
          - filter_mod: {"op":"filter_mod","mod":N,"offset":k}
          - crop: {"op":"crop","y0":..,"y1":..,"x0":..,"x1":..}
          - stride: {"op":"stride","n":N}
        """
        f = frame
        for op in pipeline:
            op_name = op.get("op")
            if op_name == "filter_mod":
                mod = int(op.get("mod", 2))
                offset = int(op.get("offset", 0))
                if (counter % mod) != offset:
                    return None
            elif op_name == "crop":
                y0 = int(op.get("y0", 0))
                y1 = int(op.get("y1", f.shape[0]))
                x0 = int(op.get("x0", 0))
                x1 = int(op.get("x1", f.shape[1]))
                f = f[y0:y1, x0:x1]
            elif op_name == "stride":
                n = int(op.get("n", 1))
                if n > 1 and (counter % n) != 0:
                    return None
            else:
                # unknown op -> ignore
                continue
        return f

    # -------------------------
    # SUB -> async-generator adapter for consumer compatibility
    # -------------------------
    def _pub_connect_addr(self) -> str:
        b = self._pub_bind_addr
        if b.startswith("tcp://*:"):
            return "tcp://127.0.0.1:" + b.split("tcp://*:")[1]
        if b.startswith("tcp://0.0.0.0:"):
            return "tcp://127.0.0.1:" + b.split("tcp://0.0.0.0:")[1]
        return b

    def producer_corofunc(self, stream_name: str) -> AsyncGenerator:
        """
        Return an async generator that yields numpy/ImageWithMetadata for the named stream
        by creating a temporary `ZmqReceiver` SUB and filtering by metadata['stream'].
        """
        async def gen():
            recv_endpoint = self._pub_bind_addr
            if recv_endpoint.startswith("tcp://*:") or recv_endpoint.startswith("tcp://0.0.0.0:"):
                recv_endpoint = self._pub_connect_addr()

            receiver = ZmqReceiver(endpoint=recv_endpoint, reliable=False, rcvhwm=10)
            try:
                async for metadata, image in receiver.subscribe(return_metadata=True):
                    if metadata is None or image is None:
                        break
                    if metadata.get("stream") != stream_name:
                        continue
                    yield image
            finally:
                try:
                    receiver.close()
                except Exception:
                    pass

        return gen()

# helper
def awaitable(obj):
    return callable(getattr(obj, "__await__", None))


# USAGE

# import asyncio
# import json
# import numpy as np
# import zmq.asyncio

# from concert.networking.base import ZmqReceiver
# from concert.helpers import ImageWithMetadata

# CONTROL_ADDR = "tcp://localhost:5555"   # producer control REP
# PUB_ADDR = "tcp://localhost:5556"       # producer PUB (pub_bind)

# async def add_stream(control_addr: str, name: str, pipeline):
#     ctx = zmq.asyncio.Context.instance()
#     req = ctx.socket(zmq.REQ)
#     req.connect(control_addr)
#     await req.send_json({"cmd": "add_stream", "name": name, "pipeline": pipeline, "backpressure": "drop_oldest"})
#     resp = await req.recv_json()
#     req.close()
#     return resp

# async def consumer_transform_loop(pub_addr: str, stream_name: str, max_frames: int = 100):
#     # Use ZmqReceiver to subscribe to producer PUB and get (metadata, ImageWithMetadata)
#     receiver = ZmqReceiver(endpoint=pub_addr, reliable=False, rcvhwm=10)
#     saved = 0
#     try:
#         async for metadata, image in receiver.subscribe(return_metadata=True):
#             if metadata is None or image is None:
#                 # end-of-stream marker
#                 break
#             if metadata.get("stream") != stream_name:
#                 continue
#             # image is an ImageWithMetadata (convertable to numpy array)
#             arr = np.asarray(image)   # or image.data depending on helper
#             # Consumer-side transform: normalise then threshold
#             a = arr.astype(np.float32)
#             a = (a - a.mean()) / (a.std() + 1e-9)
#             a = (a > 1.0).astype(np.uint8)  # example binary transform
#             # Save or forward to downstream consumer
#             np.save(f"{stream_name}_{metadata['counter']:06}.npy", a)
#             saved += 1
#             if saved >= max_frames:
#                 break
#     finally:
#         receiver.close()

# async def main():
#     stream_name = "radios_even_cropped"
#     # Configure a pipeline on the producer (declarative ops executed by producer)
#     pipeline = [
#         {"op": "filter_mod", "mod": 2, "offset": 0},            # even frames
#         {"op": "crop", "y0": 100, "y1": 900, "x0": 200, "x1": 1200}
#     ]
#     print("Adding stream on producer:", await add_stream(CONTROL_ADDR, stream_name, pipeline))
#     # Start consumer loop (will receive processed frames from the producer)
#     await consumer_transform_loop(PUB_ADDR, stream_name, max_frames=50)

# if __name__ == "__main__":
#     asyncio.run(main())
