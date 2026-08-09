import os
import signal
import threading
import time
import traceback

from dvrip import DVRIPCam
from flask import Flask, Response, abort
from src.config import config

IP, PORT = config["ip"], config["port"]
USER, PASSWORD = config["user"], config["password"]

CHANNELS = [0, 1]
SNAPSHOT_INTERVAL = 0.01

app = Flask(__name__)

state = {ch: {"frame": None, "lock": threading.Lock()} for ch in CHANNELS}
shutdown_flag = threading.Event()

cam_connections = {}

def connect_camera():
    cam = DVRIPCam(IP, user=USER, password=PASSWORD, port=PORT)
    if not cam.login():
        raise RuntimeError("Не удалось залогиниться")
    return cam


def snapshot_loop(channel):
    # у каждого канала - своё независимое соединение
    cam = connect_camera()
    cam_connections[channel] = cam
    print(f"✅ Канал {channel}: отдельное соединение установлено")

    while True:
        try:
            jpeg_bytes = cam.snapshot(channel=channel)
            if jpeg_bytes:
                with state[channel]["lock"]:
                    state[channel]["frame"] = jpeg_bytes
        except Exception as e:
            if shutdown_flag.is_set():
                break
            print(f"[канал {channel}] Ошибка снапшота: {e!r}")
            traceback.print_exc()
            try:
                cam.close()
            except Exception:
                pass
            time.sleep(2)
            try:
                cam = connect_camera()
                cam_connections[channel] = cam
                print(f"🔄 Канал {channel}: переподключились")
            except Exception as reconnect_err:
                print(f"[канал {channel}] Не удалось переподключиться: {reconnect_err!r}")
        time.sleep(SNAPSHOT_INTERVAL)


def generate_frames(channel):
    while True:
        with state[channel]["lock"]:
            frame = state[channel]["frame"]
        if frame is not None:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(SNAPSHOT_INTERVAL)


@app.route("/channel/<int:channel>")
def stream(channel):
    if channel not in CHANNELS:
        abort(404, f"Канал {channel} не найден")
    return Response(generate_frames(channel), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/shutdown", methods=["POST"])
def shutdown():
    print("🛑 Получен запрос на завершение работы...")

    shutdown_flag.set()  # сигнализируем всем потокам остановиться

    for channel, cam in cam_connections.items():
        try:
            cam.close()
            print(f"Соединение с каналом {channel} закрыто")
        except Exception as e:
            print(f"Ошибка при закрытии канала {channel}: {e}")

    def stop():
        time.sleep(0.5)
        os.kill(os.getpid(), signal.SIGINT)

    threading.Thread(target=stop, daemon=True).start()
    return "OK"


@app.route("/")
def index():
    images = "".join(
        f'''
        <div style="text-align:center; background:#1e1e1e; padding:16px; border-radius:12px;">
            <h3 style="color:#e0e0e0; font-weight:500;">Камера {ch}</h3>
            <img src="/channel/{ch}" width="480" style="border-radius:8px; border:1px solid #333;">
        </div>
        '''
        for ch in CHANNELS
    )
    return f"""
    <html>
        <head>
            <title>CCTV Stream</title>
        </head>
        <body style="
            display:flex;
            flex-direction:column;
            align-items:center;
            gap:20px;
            font-family:'Segoe UI', sans-serif;
            background:#121212;
            min-height:100vh;
            margin:0;
            padding:24px;
        ">
            <button onclick="doExit()" style="
                padding:10px 24px;
                background:#c0392b;
                color:white;
                border:none;
                border-radius:8px;
                font-size:14px;
                cursor:pointer;
            ">Выйти</button>

            <div style="display:flex; gap:20px; justify-content:center;">
                {images}
            </div>

            <script>
                function doExit() {{
                    fetch("/shutdown", {{ method: "POST" }})
                        .then(() => {{
                            document.body.innerHTML = "<h2 style='color:#e0e0e0; text-align:center; margin-top:40px;'>Соединение закрыто</h2>";
                            window.close();
                        }})
                        .catch(() => {{
                            document.body.innerHTML = "<h2 style='color:#e0e0e0; text-align:center; margin-top:40px;'>Соединение закрыто</h2>";
                            window.close();
                        }});
                }}
            </script>
        </body>
    </html>
    """


def main():
    for ch in CHANNELS:
        t = threading.Thread(target=snapshot_loop, args=(ch,), daemon=True)
        t.start()

    print("🎥 Оба канала на одной странице: http://127.0.0.1:5000/")

    app.run(host="127.0.0.1", port=5000, threaded=True)


if __name__ == "__main__":
    main()