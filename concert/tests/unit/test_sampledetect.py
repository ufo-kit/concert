import asyncio
import importlib
import inspect
import sys
import types
from unittest.mock import AsyncMock

import numpy as np
import pytest

from concert.experiments.addons.tango import SampleDetector as TangoSampleDetector

importlib.import_module("concert.imageprocessing")


@pytest.fixture
def sampledetect_module(monkeypatch):
    torch = types.ModuleType("torch")
    torch.Tensor = type("Tensor", (), {})
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    ultralytics = types.ModuleType("ultralytics")

    def make_model(path):
        return ("model", path)

    ultralytics.YOLO = make_model
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)
    module_name = "concert.ext.tangoservers.sampledetect"
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    yield module
    sys.modules.pop(module_name, None)


def unwrap(method):
    return inspect.unwrap(method)


def test_sample_detect_decodes_image_and_encodes_result(sampledetect_module):
    image = np.arange(12, dtype=np.uint16).reshape(3, 4)
    detected = []
    device = types.SimpleNamespace(
        _sample_detect=lambda value: detected.append(value.copy()) or ([1, 2, 4, 3], 0.8764)
    )

    result = unwrap(sampledetect_module.SampleDetect.sample_detect)(
        device,
        ("4/3/uint16", image.tobytes()),
    )

    np.testing.assert_array_equal(detected[0], image)
    assert result == [1, 2, 4, 3, 876]


def test_sample_detect_returns_zero_result_without_detection(sampledetect_module):
    image = np.ones((2, 3), dtype=np.float32)
    device = types.SimpleNamespace(_sample_detect=lambda value: (None, 0))

    result = unwrap(sampledetect_module.SampleDetect.sample_detect)(
        device,
        ("3/2/float32", image.tobytes()),
    )

    assert result == [0, 0, 0, 0, 0]


def test_sample_detect_model_prediction_rounds_bbox(sampledetect_module, monkeypatch):
    class Tensor:
        def __init__(self, value):
            self.value = np.asarray(value)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    box = types.SimpleNamespace(
        xyxy=[Tensor([1.8, 2.2, 8.1, 9.9])],
        conf=Tensor([0.75]),
    )

    class Prediction:
        boxes = [box]

        def __len__(self):
            return len(self.boxes)

    prediction = Prediction()
    calls = []
    model = types.SimpleNamespace(predict=lambda **kwargs: calls.append(kwargs) or [prediction])

    def debug_stream(*args):
        pass

    detector = types.SimpleNamespace(
        _model=model,
        _min_confidence=0.4,
        debug_stream=debug_stream,
    )
    converted = np.zeros((4, 5, 3), dtype=np.uint8)
    monkeypatch.setattr(sampledetect_module, "convert_image_to_nbit", lambda *a, **kw: converted)
    monkeypatch.setattr(sampledetect_module.torch.cuda, "is_available", lambda: True)

    bbox, confidence = sampledetect_module.SampleDetect._sample_detect(
        detector, np.ones((4, 5), dtype=np.uint16))

    assert bbox == [1, 2, 9, 10]
    assert confidence == pytest.approx(0.75)
    assert calls == [{
        "source": converted,
        "max_det": 1,
        "conf": 0.4,
        "device": "cuda:0",
    }]


def test_sample_detect_model_prediction_without_box(sampledetect_module, monkeypatch):
    class Prediction:
        boxes = []

        def __len__(self):
            return 0

    model = types.SimpleNamespace(predict=lambda **kwargs: [Prediction()])
    detector = types.SimpleNamespace(
        _model=model,
        _min_confidence=0.25,
        debug_stream=lambda *args: None,
    )
    monkeypatch.setattr(
        sampledetect_module,
        "convert_image_to_nbit",
        lambda *a, **kw: np.zeros((2, 2, 3), dtype=np.uint8),
    )

    assert sampledetect_module.SampleDetect._sample_detect(
        detector, np.ones((2, 2))) == (None, 0)


def test_stream_detect_sends_only_bbox_changes(sampledetect_module):
    async def subscribe():
        for value in (1, 2, 3, 4):
            yield np.array([[value]])

    results = iter([
        ([1, 2, 3, 4], 0.8),
        ([1, 2, 3, 4], 0.7),
        (None, 0),
        ([2, 3, 4, 5], 0.9),
    ])
    sender = types.SimpleNamespace(send_json=AsyncMock())
    detector = types.SimpleNamespace(
        _receiver=types.SimpleNamespace(subscribe=subscribe),
        _sender=sender,
        _sample_detect=lambda image: next(results),
        _bboxes=None,
    )

    asyncio.run(sampledetect_module.SampleDetect._stream_detect(detector))

    assert detector._bboxes == [[1, 2, 3, 4], [1, 2, 3, 4], [2, 3, 4, 5]]
    assert [call.args[0] for call in sender.send_json.await_args_list] == [
        {"sample-bbox": [1, 2, 3, 4]},
        {"sample-bbox": [0, 0, 0, 0]},
        {"sample-bbox": [2, 3, 4, 5]},
    ]


def test_maximum_rectangle_uses_percentiles(sampledetect_module):
    method = unwrap(sampledetect_module.SampleDetect.get_maximum_rectangle)
    detector = types.SimpleNamespace(_bboxes=[])
    assert asyncio.run(method(detector, 10)) == [0, 0, 0, 0]

    detector._bboxes = [
        [0, 10, 100, 110],
        [10, 20, 90, 100],
        [20, 30, 80, 90],
    ]
    assert asyncio.run(method(detector, 25)) == [5, 15, 95, 105]


def test_tango_sample_detector_encodes_image_and_confidence():
    image = np.arange(12, dtype=np.uint16).reshape(3, 4)
    device = types.SimpleNamespace(
        sample_detect=AsyncMock(return_value=[1, 2, 3, 4, 875]),
        get_maximum_rectangle=AsyncMock(return_value=[2, 3, 8, 9]),
    )
    detector = types.SimpleNamespace(_device=device)

    bbox, confidence = asyncio.run(TangoSampleDetector.detect(detector, image))

    assert bbox == [1, 2, 3, 4]
    assert confidence == pytest.approx(0.875)
    encoding, blob = device.sample_detect.await_args.args[0]
    assert encoding == "4/3/uint16"
    np.testing.assert_array_equal(np.frombuffer(blob, dtype=np.uint16).reshape(3, 4), image)
    assert asyncio.run(
        TangoSampleDetector.get_maximum_rectangle(detector, 20)) == [2, 3, 8, 9]
    device.get_maximum_rectangle.assert_awaited_once_with(20)
