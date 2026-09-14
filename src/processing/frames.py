from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Condition
from time import monotonic

import cv2
import numpy as np


@dataclass(frozen=True)
class Frame:
    channel: int
    captured_at: datetime
    received_at: float
    bgr: np.ndarray = field(repr=False)
    jpeg: bytes = field(repr=False)


def normalize(channel, raw):
    captured_at, received_at = datetime.now(UTC), monotonic()
    if isinstance(raw, bytes):
        bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        jpeg = raw
        if bgr is None:
            raise ValueError("Invalid JPEG")
    else:
        bgr = np.asarray(raw).copy()
        if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
            raise ValueError("Expected uint8 BGR image")
        ok, encoded = cv2.imencode(".jpg", bgr)
        if not ok:
            raise ValueError("JPEG encoding failed")
        jpeg = encoded.tobytes()
    bgr.flags.writeable = False
    return Frame(channel, captured_at, received_at, bgr, jpeg)


class LatestFrame:
    """One retained frame and independent sequence cursors for each consumer."""

    def __init__(self):
        self.condition = Condition()
        self.frame = None
        self.sequence = 0
        self.closed = False

    def publish(self, frame):
        with self.condition:
            if not self.closed:
                self.sequence += 1
                self.frame = frame
                self.condition.notify_all()

    def clear(self):
        with self.condition:
            self.frame = None

    def wait(self, after=0, timeout=1.0):
        with self.condition:
            self.condition.wait_for(
                lambda: (
                    self.closed or (self.frame is not None and self.sequence > after)
                ),
                timeout,
            )
            if self.closed or self.frame is None or self.sequence <= after:
                return None
            return self.sequence, self.frame

    def close(self):
        with self.condition:
            self.closed = True
            self.frame = None
            self.condition.notify_all()
