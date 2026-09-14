import sys
from dataclasses import replace
from threading import Event
from time import monotonic, sleep
from unittest.mock import Mock

import numpy as np
import pytest

from src.application.manager import CameraManager
from src.config import CameraSettings, InferenceSettings, Settings
from src.inference.base import Detection
from src.inference.pipeline import InferencePipeline, annotate
from src.inference.yolo import ModelUnavailable, YoloDetector
from src.processing.frames import LatestFrame, normalize
from src.web import create_app


def wait_until(predicate):
    deadline = monotonic() + 2
    while monotonic() < deadline:
        if predicate():
            return
        sleep(0.01)
    assert predicate()


def test_lazy_shared_model_two_cameras_and_disable():
    frames = {ch: LatestFrame() for ch in (0, 1)}
    detector = Mock()
    detector.detect.return_value = (Detection((1, 1, 6, 6), 0.9),)
    factory = Mock(return_value=detector)
    pipeline = InferencePipeline(frames, InferenceSettings(), 10, factory)
    for ch, buffer in frames.items():
        buffer.publish(normalize(ch, np.zeros((8, 8, 3), np.uint8)))
    assert pipeline.status()["state"] == "disabled"
    assert pipeline.thread is None
    factory.assert_not_called()
    try:
        pipeline.set_enabled(True)
        wait_until(lambda: pipeline.status()["cameras"] == {"0": 1, "1": 1})
        factory.assert_called_once()
        assert {call.args[0].channel for call in detector.detect.call_args_list} == {
            0,
            1,
        }
        # No repeated inference on the same frame.
        sleep(0.1)
        assert detector.detect.call_count == 2
        pipeline.set_enabled(False)
        assert pipeline.output(0) is None
        assert pipeline.status()["cameras"] == {"0": None, "1": None}
        wait_until(lambda: detector.close.call_count == 1)
    finally:
        pipeline.stop()


def test_disable_during_inference_discards_result():
    entered, release = Event(), Event()
    detector = Mock()

    def detect(frame):
        entered.set()
        assert release.wait(2)
        return ()

    detector.detect.side_effect = detect
    frames = {0: LatestFrame()}
    frames[0].publish(normalize(0, np.zeros((8, 8, 3), np.uint8)))
    pipeline = InferencePipeline(frames, InferenceSettings(), 10, lambda _: detector)
    try:
        pipeline.set_enabled(True)
        assert entered.wait(1)
        pipeline.set_enabled(False)
        release.set()
        wait_until(lambda: detector.close.called)
        assert pipeline.output(0) is None
    finally:
        release.set()
        pipeline.stop()


def test_model_error_and_retry():
    detector = Mock()
    pipeline = InferencePipeline(
        {},
        InferenceSettings(),
        10,
        Mock(side_effect=[RuntimeError("sensitive-detail"), detector]),
    )
    try:
        pipeline.set_enabled(True)
        wait_until(lambda: pipeline.status()["state"] == "error")
        assert not pipeline.status()["enabled"]
        assert "sensitive-detail" not in pipeline.status()["error"]
        pipeline.set_enabled(True)
        wait_until(lambda: pipeline.status()["state"] == "running")
    finally:
        pipeline.stop()


def test_missing_weights(tmp_path):
    with pytest.raises(ModelUnavailable):
        YoloDetector(InferenceSettings(model_path=str(tmp_path / "missing.pt")))


def test_device_default_and_validation():
    assert InferenceSettings().device == "gpu"
    assert InferenceSettings(device="cpu").device == "cpu"
    with pytest.raises(ValueError, match="ML_DEVICE"):
        InferenceSettings(device="invalid")


def test_annotation_preserves_original_and_timestamps():
    frame = normalize(0, np.zeros((32, 32, 3), np.uint8))
    result = annotate(frame, (Detection((2, 2, 20, 20), 0.8),))
    assert not frame.bgr.any()
    assert result.bgr.any()
    assert result.received_at == frame.received_at
    assert result.jpeg != frame.jpeg


def test_api_validation_and_default_disabled():
    manager = CameraManager(Settings((CameraSettings(0), CameraSettings(1))))
    client = create_app(manager).test_client()
    assert client.get("/api/detection").json["enabled"] is False
    assert client.post("/api/detection", json={"enabled": "true"}).status_code == 400
    assert (
        client.post(
            "/api/detection",
            json={"enabled": True},
            headers={"Sec-Fetch-Site": "cross-site"},
        ).status_code
        == 403
    )
    fake = Mock()
    manager.inference.factory = Mock(return_value=fake)
    try:
        assert client.post("/api/detection", json={"enabled": True}).json["enabled"]
        assert not client.post("/api/detection", json={"enabled": False}).json[
            "enabled"
        ]
        assert b"detection-toggle" in client.get("/").data
    finally:
        manager.stop()


def test_stale_output_not_displayed():
    pipeline = InferencePipeline({0: LatestFrame()}, InferenceSettings(), 1)
    frame = replace(normalize(0, np.zeros((8, 8, 3), np.uint8)), received_at=0)
    pipeline.enabled = True
    pipeline.outputs[0] = (1, frame, 2)
    assert pipeline.output(0) is None
    assert pipeline.status()["cameras"]["0"] is None
    pipeline.stop()


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_adapter_pt_and_device(tmp_path, monkeypatch, device):
    model_path = tmp_path / "model.pt"
    model_path.touch()
    torch, ultralytics, model = Mock(), Mock(), Mock()
    model.names = {0: "person", 1: "car"}
    box = Mock()
    box.cls.item.return_value = 0
    box.conf.item.return_value = 0.9
    box.xyxy = [Mock()]
    box.xyxy[0].tolist.return_value = [1, 2, 10, 20]
    result = Mock()
    result.boxes.cpu.return_value = [box]
    model.predict.return_value = [result]
    ultralytics.YOLO.return_value = model
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)
    detector = YoloDetector(
        InferenceSettings(model_path=str(model_path), device=device)
    )
    frame = normalize(0, np.zeros((160, 320, 3), np.uint8))
    assert detector.detect(frame) == (Detection((1, 2, 10, 20), 0.9),)
    args = model.predict.call_args.kwargs
    assert args["device"] == ("cpu" if device == "cpu" else "cuda:0")
    assert args["half"] == (device == "gpu")
    assert args["classes"] == [0]
    assert args["batch"] == 1
    ultralytics.YOLO.assert_called_once_with(str(model_path), task="detect")
    detector.close()
    assert detector.model is None
    assert torch.cuda.empty_cache.call_count == (device == "gpu")


def test_gpu_unavailable(tmp_path, monkeypatch):
    path = tmp_path / "model.pt"
    path.touch()
    torch, ultralytics = Mock(), Mock()
    torch.cuda.is_available.return_value = False
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)
    with pytest.raises(ModelUnavailable, match="GPU недоступен"):
        YoloDetector(InferenceSettings(model_path=str(path)))
    ultralytics.YOLO.assert_not_called()


def test_inference_rate_limit_and_latest_frame():
    frames = {0: LatestFrame()}
    detector = Mock()
    detector.detect.return_value = ()
    pipeline = InferencePipeline(
        frames, InferenceSettings(fps=1), 10, lambda _: detector
    )
    frames[0].publish(normalize(0, np.zeros((8, 8, 3), np.uint8)))
    try:
        pipeline.set_enabled(True)
        wait_until(lambda: pipeline.output(0) is not None)
        for _ in range(10):
            frames[0].publish(normalize(0, np.zeros((8, 8, 3), np.uint8)))
        sleep(0.15)
        assert detector.detect.call_count == 1
        wait_until(lambda: pipeline.output(0)[0] == 11)
        assert detector.detect.call_count == 2
    finally:
        pipeline.stop()


def test_web_uses_matching_annotated_frame_then_raw_on_disable():
    manager = CameraManager(Settings((CameraSettings(0),)))
    raw = normalize(0, np.zeros((32, 32, 3), np.uint8))
    annotated = annotate(raw, (Detection((2, 2, 20, 20), 0.9),))
    manager.frames[0].publish(raw)
    manager.inference.enabled = True
    manager.inference.outputs[0] = (1, annotated, 1)
    client = create_app(manager).test_client()
    response = client.get("/channel/0", buffered=False)
    try:
        assert annotated.jpeg in next(response.response)
        manager.inference.set_enabled(False)
        assert raw.jpeg in next(response.response)
    finally:
        response.close()
        manager.stop()
