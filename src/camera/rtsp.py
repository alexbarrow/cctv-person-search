import av


class RtspCamera:
    def __init__(self, settings, timeout):
        self.settings, self.timeout = settings, timeout
        self.container = self.frames = None

    def open(self):
        self.close()
        # Native diagnostics may include source credentials.
        av.logging.set_level(None)
        try:
            self.container = av.open(
                self.settings.stream_url,
                options={"rtsp_transport": "tcp"},
                timeout=(self.timeout, self.timeout),
            )
            self.frames = iter(self.container.decode(video=0))
        except Exception:  # noqa: BLE001 -- sanitize native exception messages
            self.close()
            raise ConnectionError("Video connection failed") from None

    def read(self):
        try:
            return next(self.frames).to_ndarray(format="bgr24")
        except Exception:  # noqa: BLE001 -- includes EOF and decoder errors
            raise ConnectionError("Video read failed") from None

    def close(self):
        container, self.container = self.container, None
        self.frames = None
        if container is not None:
            container.close()
