import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from dotenv import load_dotenv


class ConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class CameraSettings:
    channel: int
    backend: str = "dvrip"
    host: str = field(default="", repr=False)
    port: int = 34567
    user: str = field(default="", repr=False)
    password: str = field(default="", repr=False)
    stream_url: str = field(default="", repr=False)


@dataclass(frozen=True)
class InferenceSettings:
    model_path: str = "models/yolo11m.pt"
    fps: float = 2.0
    image_size: int = 640
    confidence: float = 0.15
    device: str = "gpu"

    def __post_init__(self):
        if self.device not in {"cpu", "gpu"}:
            raise ConfigurationError("ML_DEVICE must be cpu or gpu")


@dataclass(frozen=True)
class Settings:
    cameras: tuple[CameraSettings, ...]
    snapshot_interval: float = 0.2
    io_timeout: float = 5.0
    retry_initial: float = 1.0
    retry_max: float = 30.0
    stale_after: float = 10.0
    web_host: str = "localhost"
    web_port: int = 5000
    inference: InferenceSettings = field(default_factory=InferenceSettings)


def parse_settings(env: Mapping[str, str]) -> Settings:
    def required(key):
        value = env.get(key, "")
        if not value or value.startswith("<"):
            raise ConfigurationError(f"Set {key}")
        return value

    def number(key, default, integer=False):
        try:
            value = (int if integer else float)(env.get(key, str(default)))
            if not math.isfinite(value) or value <= 0:
                raise ValueError
            return value
        except (ValueError, OverflowError):
            raise ConfigurationError(
                f"Invalid {key}: expected a positive number"
            ) from None

    backend = env.get("CAMERA_BACKEND", "dvrip")
    if backend not in {"dvrip", "rtsp"}:
        raise ConfigurationError("CAMERA_BACKEND must be dvrip or rtsp")
    try:
        channels = tuple(int(v.strip()) for v in env.get("CHANNELS", "0,1").split(","))
        if min(channels) < 0 or len(set(channels)) != len(channels):
            raise ValueError
    except ValueError:
        raise ConfigurationError(
            "CHANNELS must contain unique non-negative integers"
        ) from None
    cameras = []
    for channel in channels:
        if backend == "dvrip":
            port = number("PORT", 34567, True)
            if port > 65535:
                raise ConfigurationError("Invalid PORT")
            cameras.append(
                CameraSettings(
                    channel,
                    host=required("IP"),
                    port=port,
                    user=required("USER_CAM"),
                    password=required("PASSWORD_CAM"),
                )
            )
        else:
            key = f"CAMERA_{channel}_RTSP_URL"
            url = required(key)
            try:
                parsed = urlsplit(url)
                if parsed.scheme not in {"rtsp", "rtsps"} or not parsed.hostname:
                    raise ValueError
                if parsed.port == 0:
                    raise ValueError
            except ValueError:
                raise ConfigurationError(f"Invalid {key}") from None
            cameras.append(CameraSettings(channel, backend="rtsp", stream_url=url))
    settings = Settings(
        tuple(cameras),
        number("SNAPSHOT_INTERVAL", 0.2),
        number("IO_TIMEOUT", 5),
        number("RETRY_INITIAL", 1),
        number("RETRY_MAX", 30),
        number("STALE_AFTER", 10),
        env.get("WEB_HOST", "localhost"),
        number("WEB_PORT", 5000, True),
        InferenceSettings(
            env.get("ML_MODEL_PATH", "models/yolov8n.pt"),
            number("ML_FPS", 2),
            number("ML_IMAGE_SIZE", 320, True),
            number("ML_CONFIDENCE", 0.4),
            env.get("ML_DEVICE", "gpu").strip().lower(),
        ),
    )
    if settings.retry_max < settings.retry_initial:
        raise ConfigurationError("RETRY_MAX must be >= RETRY_INITIAL")
    if settings.web_port > 65535:
        raise ConfigurationError("Invalid WEB_PORT")
    if not 0 < settings.inference.fps <= 10:
        raise ConfigurationError("ML_FPS must be <= 10")
    if settings.inference.image_size not in {320, 416, 640}:
        raise ConfigurationError("ML_IMAGE_SIZE must be 320, 416 or 640")
    if settings.inference.confidence > 1:
        raise ConfigurationError("ML_CONFIDENCE must be <= 1")
    return settings


def load_settings() -> Settings:
    load_dotenv(".env", override=False)
    return parse_settings(os.environ)
