from time import monotonic

from flask import Flask, Response, abort, render_template, request


def create_app(manager):
    app = Flask(__name__)

    @app.route("/api/detection", methods=["GET", "POST"])
    def detection():
        if request.method == "POST":
            # JSON-only same-origin controls; no cross-origin permission is granted.
            if request.headers.get("Sec-Fetch-Site") == "cross-site":
                abort(403)
            data = request.get_json(silent=True)
            if not isinstance(data, dict) or type(data.get("enabled")) is not bool:
                abort(400)
            if manager.stopping.is_set():
                abort(409)
            manager.inference.set_enabled(data["enabled"])
        return manager.inference.status(), 200, {"Cache-Control": "no-store"}

    @app.get("/")
    def index():
        return render_template("index.html", channels=tuple(manager.frames))

    @app.get("/channel/<int:channel>")
    def stream(channel):
        if channel not in manager.frames:
            abort(404)

        def generate():
            sequence = 0
            last_output = None
            while not manager.stopping.is_set():
                item = manager.frames[channel].wait(sequence, timeout=0.1)
                if item is None:
                    # A completed inference may arrive without a newer camera frame.
                    item = manager.frames[channel].wait(timeout=0)
                if item is None:
                    continue
                sequence, frame = item
                annotated = manager.inference.output(channel)
                key = ("raw", sequence)
                if annotated is not None:
                    key = ("ml", annotated[0])
                    frame = annotated[1]
                if key == last_output:
                    continue
                if monotonic() - frame.received_at <= manager.settings.stale_after:
                    last_output = key
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                        + frame.jpeg
                        + b"\r\n"
                    )

        return Response(
            generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/health")
    def health():
        fresh = {}
        for channel, buffer in manager.frames.items():
            item = buffer.wait(timeout=0)
            fresh[str(channel)] = bool(
                item
                and monotonic() - item[1].received_at <= manager.settings.stale_after
            )
        healthy = bool(fresh) and all(fresh.values()) and not manager.stopping.is_set()
        return {"healthy": healthy, "cameras": fresh}, 200 if healthy else 503

    return app
