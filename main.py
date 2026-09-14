"""Explicit startup; imports do not load secrets or connect to cameras."""

import logging
import signal

from src.application.manager import CameraManager
from src.config import ConfigurationError, load_settings
from src.web import create_app


def main():
    logging.basicConfig(level=logging.INFO)
    try:
        settings = load_settings()
    except ConfigurationError as error:
        raise SystemExit(str(error)) from None

    def terminate(signum, frame):
        raise SystemExit(0)

    previous = signal.signal(signal.SIGTERM, terminate)
    manager = CameraManager(settings)
    try:
        manager.start()
        create_app(manager).run(
            host=settings.web_host,
            port=settings.web_port,
            threaded=True,
            use_reloader=False,
        )
    finally:
        manager.stop()
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
