from dataclasses import dataclass
from typing import Protocol

from src.processing.frames import Frame


@dataclass(frozen=True)
class Detection:
    bbox: tuple[int, int, int, int]
    confidence: float


class Detector(Protocol):
    def detect(self, frame: Frame) -> tuple[Detection, ...]: ...


class NoopDetector:
    def detect(self, frame: Frame) -> tuple[Detection, ...]:
        return ()
