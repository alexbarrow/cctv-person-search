import logging
from threading import Event, Thread
from time import monotonic

from src.camera.dvrip import DvripCamera
from src.camera.rtsp import RtspCamera
from src.camera.source import FrameSource
from src.inference.pipeline import InferencePipeline
from src.processing.frames import LatestFrame, normalize

logger = logging.getLogger(__name__)


def make_source(camera, timeout) -> FrameSource:
    return (RtspCamera if camera.backend == "rtsp" else DvripCamera)(camera, timeout)


class CameraManager:
    """One-shot lifecycle; each worker exclusively owns its source."""

    def __init__(self, settings, source_factory=make_source):
        self.settings, self.source_factory = settings, source_factory
        self.frames = {camera.channel: LatestFrame() for camera in settings.cameras}
        self.inference = InferencePipeline(
            self.frames, settings.inference, settings.stale_after
        )
        self.stopping = Event()
        self.threads = []
        self.started = False

    def start(self):
        if self.started or self.stopping.is_set():
            raise RuntimeError("Manager cannot be restarted")
        self.started = True
        for camera in self.settings.cameras:
            thread = Thread(target=self._run, args=(camera,), daemon=True)
            self.threads.append(thread)
            thread.start()

    def _run(self, camera):
        buffer = self.frames[camera.channel]
        delay = self.settings.retry_initial
        while not self.stopping.is_set():
            source = None
            try:
                source = self.source_factory(camera, self.settings.io_timeout)
                source.open()
                while not self.stopping.is_set():
                    buffer.publish(normalize(camera.channel, source.read()))
                    delay = self.settings.retry_initial
                    if camera.backend == "dvrip":
                        self.stopping.wait(self.settings.snapshot_interval)
            except Exception:  # noqa: BLE001 -- isolate source failures; never log secrets
                buffer.clear()
                if not self.stopping.is_set():
                    logger.warning("Camera %s unavailable; retrying", camera.channel)
            finally:
                if source is not None:
                    try:
                        source.close()
                    except Exception:  # noqa: BLE001 -- cleanup must not prevent retries
                        logger.warning("Camera %s cleanup failed", camera.channel)
            if self.stopping.wait(delay):
                break
            delay = min(delay * 2, self.settings.retry_max)

    def stop(self):
        self.stopping.set()
        for buffer in self.frames.values():
            buffer.close()
        deadline = monotonic() + self.settings.io_timeout * 3 + 1
        for thread in self.threads:
            thread.join(max(0, deadline - monotonic()))
        self.inference.stop()
        if any(thread.is_alive() for thread in self.threads):
            raise RuntimeError("Camera worker did not stop within shutdown deadline")
