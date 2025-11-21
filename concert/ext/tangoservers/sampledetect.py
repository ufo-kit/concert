"""
Tango server for benchmarking zmq transfers.
"""
import numpy as np
import torch
import sys
from tango import CmdArgType, DebugIt, InfoIt
from tango.server import attribute, AttrWriteType, command
from .base import TangoRemoteProcessing
from concert.networking.base import ZmqSender
try:
    from ultralytics import YOLO
except ImportError:
    print("You must install ultralytics to use sample detection", file=sys.stderr)


class SampleDetect(TangoRemoteProcessing):
    """
    Device server for elmo_main.py-controller based
    """

    model_path = attribute(
        label="AI model path for sample detection",
        dtype=str,
        access=AttrWriteType.READ_WRITE,
        fget="get_model_path",
        fset="set_model_path"
    )

    min_confidence = attribute(
        label="Minimum confidence for succesful detection [0 - 1]",
        dtype=float,
        access=AttrWriteType.READ_WRITE,
        fget="get_min_confidence",
        fset="set_min_confidence"
    )

    sending_port = attribute(
        label="Port over which detected images will be sent",
        dtype=int,
        access=AttrWriteType.READ_WRITE,
        fget="get_sending_port",
        fset="set_sending_port"
    )

    async def init_device(self):
        await super().init_device()
        self._model_path = ""
        self._model = None
        self._min_confidence = 0.25
        self._sender = None
        self._sending_port = 0

    @InfoIt()
    async def get_model_path(self):
        """Get current model path."""
        return self._model_path

    @InfoIt(show_args=True)
    async def set_model_path(self, path):
        """Set model path."""
        self._model_path = path
        self._model = YOLO(path)

    @InfoIt()
    async def get_min_confidence(self):
        """Get minimum confidence for detection."""
        return self._min_confidence

    @InfoIt(show_args=True)
    async def set_min_confidence(self, confidence):
        """Set minimum confidence for detection."""
        self._min_confidence = confidence

    @InfoIt()
    async def get_sending_port(self):
        """Get port over which images will be forwarded."""
        return self._sending_port

    @InfoIt(show_args=True)
    async def set_sending_port(self, port):
        """Set port over which images will be forwarded."""
        self._sending_port = port
        self._sender = await ZmqSender(endpoint=f"tcp://*:{port}", reliable=False, sndhwm=1)

    @DebugIt()
    @command()
    async def stop_sending(self):
        if self._sender:
            await self._sender.close()

    @command(dtype_out=bool)
    def is_cuda_available(self):
        return torch.cuda.is_available()

    @DebugIt(show_args=True)
    @command(dtype_in=CmdArgType.DevEncoded, dtype_out=[int, int, int, int])
    def sample_detect(self, data):
        encoding, image = data
        width, height, dtype = encoding.split("/")
        image = np.frombuffer(image, dtype=dtype).reshape(int(height), int(width))
        bbox = self._sample_detect(image)
        if bbox is None:
            bbox = [0, 0, 0, 0]

        return bbox

    def _sample_detect(self, image):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        result = self._model.predict(
            source=_grayscale_to_rgb(image), max_det=1, conf=self._min_confidence, device=device
        )
        if len(result[0]):
            result = result[0].boxes[0]
            box = result.xyxy[0].detach().cpu().numpy()
            # Round x0 and y0 down and x1, y1 up
            bbox = [
                int(np.floor(box[0])),
                int(np.floor(box[1])),
                int(np.ceil(box[2])),
                int(np.ceil(box[3]))
            ]
            confidence = result.conf.detach().cpu().numpy()[0]
        else:
            bbox = None
            confidence = 0

        self.debug_stream("sample at: %s, confidence: %.3f", bbox, confidence)

        return bbox

    async def _stream_detect(self):
        self._bboxes = []
        last = None

        async for image in self._receiver.subscribe():
            bbox = self._sample_detect(image)
            if bbox is None:
                bbox = [0, 0, 0, 0]
            else:
                self._bboxes.append(bbox)
            if self._sender and last != bbox:
                print("send", bbox)
                await self._sender.send_json({"sample-bbox": bbox})
            last = bbox

    @DebugIt()
    @command()
    async def stream_detect(self):
        await self._process_stream(self._stream_detect())

    @DebugIt(show_ret=True)
    @command(dtype_out=[int, int, int, int])
    async def get_maximum_rectangle(self):
        if self._bboxes == []:
            bbox = [0, 0, 0, 0]
        else:
            bboxes = np.array(self._bboxes)
            bbox = [
                np.min(bboxes[:, 0]),
                np.min(bboxes[:, 1]),
                np.max(bboxes[:, 2]),
                np.max(bboxes[:, 3]),
            ]

        return bbox


def _grayscale_to_rgb(image, percentile=0.1):
    lower = np.percentile(image, percentile)
    upper = np.percentile(image, 100 - percentile)

    img_clipped = np.clip(image, lower, upper)
    img_8bit = (((img_clipped - lower) / (upper - lower)) * 255).astype(np.uint8)

    return np.dstack((img_8bit,) * 3)
