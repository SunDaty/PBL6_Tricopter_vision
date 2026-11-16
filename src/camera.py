# src/camera.py
import argparse
import threading
import time
from typing import Optional
import cv2
import numpy as np


class BaseCamera:
    """Giao diện chuẩn cho mọi nguồn Camera (Gazebo Sim, Raspberry Pi Camera v3, Mock)."""
    last_capture_time: Optional[float] = None  # monotonic (s) lúc ảnh được chụp/nhận

    def start(self):
        pass

    def read_frame(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def stop(self):
        pass


class GazeboCamera(BaseCamera):
    """
    Nguồn nhận ảnh từ Gazebo Transport (dùng trong chế độ mô phỏng --mode sim).
    Import thư viện gz.transport13 bên trong hàm start() để khi chạy trên Raspberry Pi 5 không cần cài Gazebo.
    """
    def __init__(self, topic: str = "/camera/image_raw", width: int = 640, height: int = 480):
        self.topic = topic
        self.width = width
        self.height = height
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._has_new_frame = False
        self._latest_ts: Optional[float] = None
        self._gz_node = None

    def start(self):
        try:
            from gz.transport13 import Node
            from gz.msgs10.image_pb2 import Image
        except ImportError as exc:
            raise RuntimeError(
                "Không tìm thấy thư viện gz.transport13. "
                "Nếu đang chạy trên Raspberry Pi 5, hãy dùng '--mode real'."
            ) from exc

        self._gz_node = Node()
        if not self._gz_node.subscribe(Image, self.topic, self._on_image):
            raise RuntimeError(f"Không thể subscribe vào topic Gazebo: {self.topic}")
        print(f"[*] [GazeboCamera] Đã đăng ký nhận luồng camera từ topic: {self.topic}")

    def _on_image(self, msg):
        h, w = int(msg.height), int(msg.width)
        if h <= 0 or w <= 0 or len(msg.data) != h * w * 3:
            return
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        frame_rgb = buf.reshape((h, w, 3))
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        with self._lock:
            self._latest_frame = frame_bgr
            self._latest_ts = time.monotonic()
            self._has_new_frame = True

    def read_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if not self._has_new_frame or self._latest_frame is None:
                return None
            frame = self._latest_frame.copy()
            self.last_capture_time = self._latest_ts
            self._has_new_frame = False
        if (frame.shape[1], frame.shape[0]) != (self.width, self.height):
            frame = cv2.resize(frame, (self.width, self.height))
        return frame


class PiCameraV3(BaseCamera):
    """
    Nguồn nhận ảnh trực tiếp từ Raspberry Pi Camera Module v3 (IMX708) trên Raspberry Pi 5
    sử dụng thư viện Picamera2 (libcamera).
    """
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self._picam2 = None

    def start(self):
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError(
                "Không tìm thấy thư viện 'picamera2'. "
                "Trên Raspberry Pi 5, cài đặt bằng lệnh: sudo apt install -y python3-picamera2"
            ) from exc

        self._picam2 = Picamera2()
        config = self._picam2.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            controls={"FrameRate": self.fps},
        )
        self._picam2.configure(config)
        self._picam2.start()

        # Bật lấy nét tự động liên tục (Continuous Autofocus) trên Camera Module v3
        try:
            self._picam2.set_controls({"AfMode": 2})
        except Exception:
            pass

        print(f"[*] [PiCameraV3] Đã khởi động Camera Module v3 ({self.width}x{self.height} @ {self.fps}fps)")

    def read_frame(self) -> Optional[np.ndarray]:
        if self._picam2 is None:
            return None
        request = self._picam2.capture_request()
        try:
            frame = request.make_array("main")
            ts_ns = request.get_metadata().get("SensorTimestamp")
        finally:
            request.release()
        now = time.monotonic()
        self.last_capture_time = now
        if ts_ns is not None:
            # SensorTimestamp theo CLOCK_BOOTTIME -> quy về monotonic
            try:
                off = time.monotonic_ns() - time.clock_gettime_ns(time.CLOCK_BOOTTIME)
                cap = (ts_ns + off) / 1e9
                if 0.0 <= now - cap < 1.0:
                    self.last_capture_time = cap
            except Exception:
                pass
        if frame is None:
            return None
        if (frame.shape[1], frame.shape[0]) != (self.width, self.height):
            frame = cv2.resize(frame, (self.width, self.height))
        return np.ascontiguousarray(frame)

    def stop(self):
        if self._picam2 is not None:
            self._picam2.stop()
            self._picam2.close()
            self._picam2 = None


class MockCamera(BaseCamera):
    """Camera giả lập có người di chuyển để kiểm thử nhanh không cần phần cứng."""
    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self._step = 0

    def start(self):
        print(f"[*] [MockCamera] Đã khởi tạo camera giả lập ({self.width}x{self.height})")

    def read_frame(self) -> Optional[np.ndarray]:
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.last_capture_time = time.monotonic()
        cx = int(self.width / 2 + 120 * np.sin(self._step * 0.1))
        self._step += 1
        cv2.rectangle(frame, (cx - 40, 140), (cx + 40, 360), (200, 200, 200), -1)
        return frame


def create_camera(
    mode: str = "sim",
    image_topic: str = "/camera/image_raw",
    width: int = 640,
    height: int = 480,
) -> BaseCamera:
    """Khởi tạo nguồn camera theo chế độ ('sim', 'real', hoặc 'mock')."""
    mode = mode.lower()
    if mode == "sim":
        return GazeboCamera(topic=image_topic, width=width, height=height)
    elif mode == "real":
        return PiCameraV3(width=width, height=height)
    elif mode == "mock":
        return MockCamera(width=width, height=height)
    else:
        raise ValueError(f"Chế độ camera không hợp lệ: {mode}. Chọn 'sim', 'real' hoặc 'mock'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiểm thử độc lập Module Camera")
    parser.add_argument("--mode", choices=["sim", "real", "mock"], default="mock")
    parser.add_argument("--topic", default="/camera/image_raw")
    parser.add_argument("--test-mock", action="store_true", help="Chạy tự kiểm thử nhanh với MockCamera")
    args = parser.parse_args()

    print("=== Kiểm thử độc lập Module 2: camera.py ===")
    cam_mode = "mock" if args.test_mock else args.mode
    cam = create_camera(mode=cam_mode, image_topic=args.topic)
    cam.start()
    frame = cam.read_frame()
    assert frame is not None and frame.shape == (480, 640, 3)
    cam.stop()
    print(f"[PASS] Đọc frame thành công từ {cam.__class__.__name__}: shape={frame.shape}")
    print("=== Module camera.py hoạt động tốt! ===")
