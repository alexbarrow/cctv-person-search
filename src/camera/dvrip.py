import socket

from dvrip import DVRIPCam


class DvripCamera:
    def __init__(self, settings, timeout):
        self.settings, self.timeout = settings, timeout
        self.cam = None

    def open(self):
        self.close()
        cfg = self.settings
        cam = self.cam = DVRIPCam(
            cfg.host, user=cfg.user, password=cfg.password, port=cfg.port
        )
        try:
            # Library connect sets its timeout too late; establish the socket here.
            cam.socket = socket.create_connection(
                (cfg.host, cfg.port), timeout=self.timeout
            )
            cam.timeout = self.timeout
            cam.socket_send, cam.socket_recv = cam.tcp_socket_send, cam.tcp_socket_recv
            if not cam.login():
                raise ConnectionError("Camera login failed")
        except Exception:  # noqa: BLE001 -- sanitize third-party exception messages
            self.close()
            raise ConnectionError("Camera connection failed") from None

    def read(self):
        frame = self.cam.snapshot(channel=self.settings.channel)
        if not frame:
            raise ConnectionError("Empty camera frame")
        return bytes(frame)

    def close(self):
        cam, self.cam = self.cam, None
        if cam is not None:
            sock = cam.socket
            try:
                cam.close()
            finally:
                # Library close can skip the socket when login failed.
                if sock is not None:
                    sock.close()
