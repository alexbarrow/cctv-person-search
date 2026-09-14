"""One lazy worker/model shared fairly across all channels."""

import logging
from dataclasses import replace
from threading import Condition, Thread
from time import monotonic

import cv2

from src.inference.yolo import ModelUnavailable, YoloDetector

logger = logging.getLogger(__name__)


def annotate(frame, detections):
    image = frame.bgr.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (127, 163, 16), 2)
        cv2.putText(
            image,
            f"person {detection.confidence:.0%}",
            (x1, max(14, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (127, 163, 16),
            1,
        )
    ok, jpeg = cv2.imencode(".jpg", image)
    if not ok:
        raise ValueError("Annotation encoding failed")
    image.flags.writeable = False
    return replace(frame, bgr=image, jpeg=jpeg.tobytes())


class InferencePipeline:
    def __init__(self, frames, settings, stale_after, detector_factory=YoloDetector):
        self.frames, self.settings = frames, settings
        self.stale_after, self.factory = stale_after, detector_factory
        self.condition = Condition()
        self.enabled = self.closed = False
        self.generation = 0
        self.state, self.error = "disabled", None
        self.outputs = {}
        self.thread = None

    def set_enabled(self, enabled):
        with self.condition:
            if self.closed:
                raise RuntimeError("Pipeline stopped")
            if enabled == self.enabled:
                return
            self.enabled = enabled
            self.generation += 1
            self.outputs.clear()
            self.error = None
            self.state = "loading" if enabled else "disabled"
            if enabled and self.thread is None:
                self.thread = Thread(
                    target=self._run, daemon=True, name="person-detection"
                )
                self.thread.start()
            self.condition.notify_all()

    def status(self):
        with self.condition:
            return {
                "enabled": self.enabled,
                "state": self.state,
                "error": self.error,
                "cameras": {str(ch): self._count(ch) for ch in self.frames},
            }

    def _count(self, channel):
        item = self.outputs.get(channel)
        return (
            item[2]
            if item and monotonic() - item[1].received_at <= self.stale_after
            else None
        )

    def output(self, channel):
        with self.condition:
            item = self.outputs.get(channel)
            if (
                self.enabled
                and item
                and monotonic() - item[1].received_at <= self.stale_after
            ):
                return item[:2]
            return None

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or self.enabled)
                if self.closed:
                    return
                generation = self.generation
            detector = None
            try:
                detector = self.factory(self.settings)
                seen, due = {}, {}
                with self.condition:
                    if self.generation == generation:
                        self.state = "running"
                while True:
                    with self.condition:
                        if (
                            self.closed
                            or not self.enabled
                            or self.generation != generation
                        ):
                            break
                    for channel, buffer in self.frames.items():
                        with self.condition:
                            if (
                                self.closed
                                or not self.enabled
                                or self.generation != generation
                            ):
                                break
                        item = buffer.wait(timeout=0)
                        if (
                            item is None
                            or monotonic() - item[1].received_at > self.stale_after
                        ):
                            with self.condition:
                                self.outputs.pop(channel, None)
                            continue
                        sequence, frame = item
                        if sequence == seen.get(channel) or monotonic() < due.get(
                            channel, 0
                        ):
                            continue
                        seen[channel] = sequence
                        due[channel] = monotonic() + 1 / self.settings.fps
                        detections = detector.detect(frame)
                        result = annotate(frame, detections)
                        with self.condition:
                            if (
                                self.enabled
                                and self.generation == generation
                                and not self.closed
                            ):
                                self.outputs[channel] = (
                                    sequence,
                                    result,
                                    len(detections),
                                )
                    with self.condition:
                        self.condition.wait_for(
                            lambda generation=generation: (
                                self.closed or self.generation != generation
                            ),
                            timeout=0.05,
                        )
            except Exception as error:  # noqa: BLE001 -- isolate model failure from video
                with self.condition:
                    if self.generation == generation:
                        self.enabled = False
                        self.state = "error"
                        self.error = (
                            str(error)
                            if isinstance(error, ModelUnavailable)
                            else "Не удалось выполнить детекцию. Проверьте PT-модель и устройство."
                        )
                        self.outputs.clear()
            finally:
                if detector is not None:
                    try:
                        detector.close()
                    except Exception:  # noqa: BLE001 -- preserve lifecycle on cleanup error
                        logger.warning("Inference cleanup failed")

    def stop(self):
        with self.condition:
            self.closed = True
            self.enabled = False
            self.generation += 1
            self.state = "disabled"
            self.outputs.clear()
            self.condition.notify_all()
        if self.thread is not None:
            self.thread.join(10)
            if self.thread.is_alive():
                raise RuntimeError("Inference worker did not stop within deadline")
