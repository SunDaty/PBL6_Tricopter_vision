# src/web_server.py
import logging
import threading
from typing import Callable, Optional
import cv2
from flask import Flask, Response, jsonify, request
import numpy as np
from werkzeug.serving import make_server


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Tricopter Vision Tracking</title>
  <style>
    body {
      background-color: #121212;
      color: #f0f0f0;
      font-family: monospace;
      text-align: center;
      margin: 16px;
    }
    h2 { margin-bottom: 6px; color: #7CFC00; }
    .hint { color: #b5b5b5; font-size: 13px; margin-bottom: 12px; }
    .stream-wrap { display: inline-block; position: relative; }
    #videoFeed {
      max-width: 100%;
      height: auto;
      border: 2px solid #00ff66;
      border-radius: 6px;
      cursor: crosshair;
      box-shadow: 0 0 15px rgba(0, 255, 102, 0.25);
    }
    .controls { margin-top: 12px; display: flex; justify-content: center; gap: 12px; align-items: center; }
    button {
      background-color: #d9534f;
      color: #fff;
      border: none;
      padding: 8px 16px;
      border-radius: 4px;
      cursor: pointer;
      font-family: monospace;
      font-size: 14px;
    }
    button:hover { background-color: #c9302c; }
    #statusText { color: #00e5ff; font-size: 13px; }
  </style>
</head>
<body>
  <h2>Tricopter Vision Tracking (RPi 5)</h2>
  <div class="hint">Chạm / Click vào người trên video để khóa mục tiêu</div>
  <div class="stream-wrap">
    <img id="videoFeed" src="/video_feed" width="__W__" height="__H__" alt="Video Stream" />
  </div>
  <div class="controls">
    <button id="resetBtn" type="button">Reset / Hover</button>
    <span id="statusText">Sẵn sàng</span>
  </div>
  <script>
    const img = document.getElementById("videoFeed");
    const statusText = document.getElementById("statusText");
    const frameW = __W__;
    const frameH = __H__;

    img.addEventListener("click", async (evt) => {
      const rect = img.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return;
      const x = Math.round(((evt.clientX - rect.left) / rect.width) * frameW);
      const y = Math.round(((evt.clientY - rect.top) / rect.height) * frameH);
      try {
        const res = await fetch("/select_target", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ x, y })
        });
        const data = await res.json();
        statusText.textContent = data.message || ("Click (" + x + ", " + y + ")");
      } catch (err) {
        statusText.textContent = "Lỗi gửi lệnh chọn mục tiêu";
      }
    });

    document.getElementById("resetBtn").addEventListener("click", async () => {
      try {
        const res = await fetch("/reset_target", { method: "POST" });
        const data = await res.json();
        statusText.textContent = data.message || "Đã hủy khóa -> Hover";
      } catch (err) {
        statusText.textContent = "Lỗi gửi lệnh Reset";
      }
    });
  </script>
</body>
</html>
"""


class TrackingWebServer:
    """
    Web Server MJPEG Stream dùng Flask tối ưu chống sụt áp trên Raspberry Pi 5:
    - Chỉ nén ảnh JPEG (cv2.imencode) khi thực sự có trình duyệt đang kết nối (active_clients > 0).
    - Hỗ trợ click/chạm trực tiếp lên video trên trình duyệt để chọn hoặc hủy khóa mục tiêu.
    """
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5000,
        frame_width: int = 640,
        frame_height: int = 480,
        jpeg_quality: int = 60,
        on_select_point: Optional[Callable[[int, int], Optional[int]]] = None,
        on_reset_target: Optional[Callable[[], None]] = None,
    ):
        self.host = host
        self.port = port
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.jpeg_quality = jpeg_quality
        self.on_select_point = on_select_point
        self.on_reset_target = on_reset_target

        self._condition = threading.Condition()
        self._latest_jpeg: Optional[bytes] = None
        self._frame_id: int = 0
        self._active_clients: int = 0
        self._stopping = False
        self._encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

        self.app = self._create_app()
        self._server = None
        self._thread: Optional[threading.Thread] = None

    @property
    def has_clients(self) -> bool:
        with self._condition:
            return self._active_clients > 0

    def _create_app(self) -> Flask:
        app = Flask(__name__)
        logging.getLogger("werkzeug").setLevel(logging.ERROR)

        @app.route("/")
        def index():
            html = (
                _HTML_TEMPLATE
                .replace("__W__", str(self.frame_width))
                .replace("__H__", str(self.frame_height))
            )
            return html

        @app.route("/video_feed")
        def video_feed():
            return Response(
                self._generate_mjpeg(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @app.route("/select_target", methods=["POST"])
        def select_target():
            payload = request.get_json(silent=True) or {}
            try:
                x = max(0, min(self.frame_width - 1, int(payload.get("x", -1))))
                y = max(0, min(self.frame_height - 1, int(payload.get("y", -1))))
            except (TypeError, ValueError):
                return jsonify({"ok": False, "message": "Tọa độ không hợp lệ"}), 400

            selected_id = None
            if self.on_select_point is not None:
                selected_id = self.on_select_point(x, y)

            if selected_id is not None:
                return jsonify({"ok": True, "selected_id": selected_id, "message": f"Đã khóa ID #{selected_id}"})
            return jsonify({"ok": False, "selected_id": None, "message": f"Không có người tại ({x}, {y})"})

        @app.route("/reset_target", methods=["POST"])
        def reset_target():
            if self.on_reset_target is not None:
                self.on_reset_target()
            return jsonify({"ok": True, "message": "Đã hủy khóa -> Hover"})

        return app

    def _generate_mjpeg(self):
        last_sent_id = -1
        with self._condition:
            self._active_clients += 1
        try:
            while not self._stopping:
                with self._condition:
                    while (
                        not self._stopping
                        and (self._latest_jpeg is None or self._frame_id == last_sent_id)
                    ):
                        self._condition.wait(timeout=1.0)
                    if self._stopping:
                        break
                    if self._latest_jpeg is None or self._frame_id == last_sent_id:
                        continue
                    frame_bytes = self._latest_jpeg
                    last_sent_id = self._frame_id

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
        finally:
            with self._condition:
                self._active_clients = max(0, self._active_clients - 1)

    def publish_frame(self, frame_bgr: np.ndarray) -> bool:
        """Chỉ nén JPEG và thông báo tới các client nếu có ít nhất 1 trình duyệt đang kết nối."""
        if not self.has_clients:
            return False
        ok, encoded = cv2.imencode(".jpg", frame_bgr, self._encode_params)
        if not ok:
            return False
        jpeg_bytes = encoded.tobytes()
        with self._condition:
            self._latest_jpeg = jpeg_bytes
            self._frame_id += 1
            self._condition.notify_all()
        return True

    def start(self):
        if self._thread is not None:
            return
        self._stopping = False
        self._server = make_server(self.host, self.port, self.app, threaded=True)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f"[*] [WebServer] Đang phát MJPEG Stream tại: http://<IP>:{self.port}")

    def stop(self):
        self._stopping = True
        with self._condition:
            self._condition.notify_all()
        if self._server is not None:
            self._server.shutdown()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None


if __name__ == "__main__":
    print("=== Kiểm thử độc lập: web_server.py (TrackingWebServer) ===")
    selected_points = []
    reset_calls = []

    web = TrackingWebServer(
        on_select_point=lambda x, y: (selected_points.append((x, y)), 42)[1],
        on_reset_target=lambda: reset_calls.append(True),
    )

    # 1. Không có client kết nối -> publish_frame không nén JPEG
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    assert web.publish_frame(dummy) is False
    print("[PASS] 1. Tiết kiệm CPU: Không nén JPEG khi active_clients == 0")

    # 2. Kiểm thử HTTP routes qua Flask test_client
    client = web.app.test_client()
    r_index = client.get("/")
    assert r_index.status_code == 200 and b"/video_feed" in r_index.data
    r_sel = client.post("/select_target", json={"x": 320, "y": 240})
    assert r_sel.status_code == 200 and r_sel.get_json()["selected_id"] == 42
    assert selected_points == [(320, 240)]
    r_rst = client.post("/reset_target")
    assert r_rst.status_code == 200 and len(reset_calls) == 1
    print("[PASS] 2. HTTP routes ('/', '/select_target', '/reset_target') hoạt động đúng")
    print("=== Module web_server.py hoạt động tốt! ===")
