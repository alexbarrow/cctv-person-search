from dataclasses import replace
from threading import Event
from unittest.mock import Mock

import numpy as np
import pytest

from src.application.manager import CameraManager
from src.camera.dvrip import DvripCamera
from src.camera.rtsp import RtspCamera
from src.config import CameraSettings, ConfigurationError, Settings, parse_settings
from src.processing.frames import LatestFrame, normalize
from src.web import create_app


@pytest.fixture
def frame():
    return normalize(0, np.zeros((8, 8, 3), dtype=np.uint8))


@pytest.mark.parametrize(
    "values,key",
    [
        ({}, "IP"),
        ({"CHANNELS": "0,0"}, "CHANNELS"),
        ({"CHANNELS": ""}, "CHANNELS"),
        ({"PORT": "bad"}, "PORT"),
        ({"PORT": "65536"}, "PORT"),
        ({"CAMERA_BACKEND": "bad"}, "CAMERA_BACKEND"),
        ({"CAMERA_BACKEND": "rtsp"}, "CAMERA_0_RTSP_URL"),
    ],
)
def test_invalid_settings(values, key):
    with pytest.raises(ConfigurationError, match=key):
        parse_settings(values)


def test_normalization(frame):
    decoded = normalize(0, frame.jpeg)
    assert decoded.bgr.shape == (8, 8, 3)
    assert not decoded.bgr.flags.writeable
    assert decoded.jpeg == frame.jpeg
    with pytest.raises(ValueError):
        normalize(0, b"invalid")


def test_latest_frame_independent_consumers_and_close(frame):
    buffer = LatestFrame()
    buffer.publish(frame)
    buffer.publish(frame)
    assert buffer.wait(0, 0) == buffer.wait(0, 0) == (2, frame)
    assert buffer.wait(2, 0) is None
    buffer.clear()
    assert buffer.wait(0, 0) is None
    buffer.publish(frame)
    assert buffer.wait(2, 0)[0] == 3
    buffer.close()
    assert buffer.wait(0, 0) is None


def test_initial_failure_read_failure_reconnect_and_stop(frame, caplog):
    sources = []
    settings = Settings(
        (CameraSettings(0),), snapshot_interval=0.01, retry_initial=0.01, retry_max=0.02
    )

    def factory(camera, timeout):
        source = Mock()
        index = len(sources)
        if index == 0:
            source.open.side_effect = RuntimeError("sensitive-detail")
        elif index == 1:
            source.read.side_effect = RuntimeError("sensitive-detail")
        else:
            source.read.return_value = frame.jpeg
        sources.append(source)
        return source

    manager = CameraManager(settings, factory)
    manager.start()
    try:
        assert manager.frames[0].wait(timeout=2) is not None
    finally:
        manager.stop()
    assert len(sources) == 3
    assert all(source.close.call_count == 1 for source in sources)
    assert all(not thread.is_alive() for thread in manager.threads)
    assert "sensitive-detail" not in caplog.text
    manager.stop()
    with pytest.raises(RuntimeError):
        manager.start()


def test_stop_interrupts_backoff():
    attempted = Event()
    source = Mock()

    def fail():
        attempted.set()
        raise ConnectionError

    source.open.side_effect = fail
    manager = CameraManager(
        Settings((CameraSettings(0),), retry_initial=30), lambda *args: source
    )
    manager.start()
    assert attempted.wait(1)
    manager.stop()
    assert not manager.threads[0].is_alive()


def test_web_routes_freshness_and_no_camera_side_effect(frame):
    factory = Mock()
    manager = CameraManager(Settings((CameraSettings(0),)), factory)
    client = create_app(manager).test_client()
    assert client.get("/").status_code == 200
    assert client.get("/channel/99").status_code == 404
    assert client.post("/shutdown").status_code == 404
    assert client.get("/health").status_code == 503
    manager.frames[0].publish(frame)
    assert client.get("/health").status_code == 200
    response = client.get("/channel/0", buffered=False)
    assert frame.jpeg in next(response.response)
    response.close()
    manager.frames[0].publish(replace(frame, received_at=0))
    assert client.get("/health").status_code == 503
    factory.assert_not_called()
    manager.stop()


def test_dvr_failed_login_closes_socket(monkeypatch):
    cam, sock = Mock(), Mock()
    cam.login.return_value = False
    monkeypatch.setattr("src.camera.dvrip.DVRIPCam", Mock(return_value=cam))
    connect = Mock(return_value=sock)
    monkeypatch.setattr("src.camera.dvrip.socket.create_connection", connect)
    source = DvripCamera(CameraSettings(0), 0.5)
    with pytest.raises(ConnectionError):
        source.open()
    assert connect.call_args.kwargs["timeout"] == 0.5
    sock.close.assert_called_once()
    source.close()
    sock.close.assert_called_once()


def test_rtsp_decode_eof_and_cleanup(monkeypatch):
    container = Mock()
    video_frame = Mock()
    video_frame.to_ndarray.return_value = np.zeros((8, 8, 3), dtype=np.uint8)
    container.decode.return_value = iter([video_frame])
    opener = Mock(return_value=container)
    monkeypatch.setattr("src.camera.rtsp.av.open", opener)
    source = RtspCamera(CameraSettings(0, backend="rtsp"), 0.5)
    source.open()
    assert opener.call_args.kwargs["timeout"] == (0.5, 0.5)
    assert source.read().shape == (8, 8, 3)
    with pytest.raises(ConnectionError, match="Video read failed"):
        source.read()
    source.close()
    source.close()
    container.close.assert_called_once()


def test_rtsp_partial_open_cleanup(monkeypatch):
    container = Mock()
    container.decode.side_effect = RuntimeError("sensitive-detail")
    monkeypatch.setattr("src.camera.rtsp.av.open", Mock(return_value=container))
    source = RtspCamera(CameraSettings(0, backend="rtsp"), 0.5)
    with pytest.raises(ConnectionError, match="^Video connection failed$"):
        source.open()
    container.close.assert_called_once()


def test_retry_delay_is_bounded():
    camera = CameraSettings(0)
    source = Mock()
    source.open.side_effect = ConnectionError
    manager = CameraManager(
        Settings((camera,), retry_initial=1, retry_max=3), lambda *args: source
    )
    stopping = Mock()
    stopping.is_set.return_value = False
    stopping.wait.side_effect = [False, False, False, True]
    manager.stopping = stopping
    manager._run(camera)
    assert [call.args[0] for call in stopping.wait.call_args_list] == [1, 2, 3, 3]
    assert source.close.call_count == 4
