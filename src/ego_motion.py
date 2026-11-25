# src/ego_motion.py
"""
Công cụ bù chuyển động góc quay camera (Ego-Motion Compensation) và đồng bộ lịch sử góc Yaw
từ Pixhawk 6C / ArduPilot phục vụ ByteTrack và điều khiển bám mục tiêu có bù độ trễ.
"""
from collections import deque
import math
import threading
import time
from typing import Optional
import numpy as np


def wrap_angle_rad(angle_rad: float) -> float:
    """Đưa góc (rad) về khoảng [-pi, +pi)."""
    return ((angle_rad + math.pi) % (2.0 * math.pi)) - math.pi


class YawHistory:
    """
    Bộ đệm vòng thread-safe lưu lịch sử (t_monotonic, yaw_unwrapped_rad, yawspeed_rad_s)
    nhận từ luồng MAVLink ATTITUDE (30-50 Hz).
    """
    def __init__(self, keep_s: float = 3.0):
        self.keep_s = max(0.5, float(keep_s))
        self._buf: deque[tuple[float, float, float]] = deque()
        self._lock = threading.Lock()
        self._last_raw_yaw: Optional[float] = None
        self._last_unwrapped_yaw: float = 0.0

    def clear(self):
        with self._lock:
            self._buf.clear()
            self._last_raw_yaw = None
            self._last_unwrapped_yaw = 0.0

    def add(self, t: float, yaw_rad: float, yawspeed_rad: float = 0.0):
        """Thêm một mẫu ATTITUDE mới, tự động unwrap góc yaw qua biên ±pi."""
        t = float(t)
        yaw_rad = float(yaw_rad)
        yawspeed_rad = float(yawspeed_rad)
        with self._lock:
            if self._last_raw_yaw is None:
                unwrapped = yaw_rad
            else:
                delta = wrap_angle_rad(yaw_rad - self._last_raw_yaw)
                unwrapped = self._last_unwrapped_yaw + delta
            self._last_raw_yaw = yaw_rad
            self._last_unwrapped_yaw = unwrapped

            if self._buf and t < self._buf[-1][0]:
                # Giữ thứ tự thời gian tăng dần
                return
            self._buf.append((t, unwrapped, yawspeed_rad))
            cutoff = t - self.keep_s
            while len(self._buf) > 2 and self._buf[0][0] < cutoff:
                self._buf.popleft()

    def is_fresh(self, now: Optional[float] = None, max_age_s: float = 0.2) -> bool:
        """Kiểm tra có dữ liệu yaw gần đây (trong vòng max_age_s) hay không."""
        cur = now if now is not None else time.monotonic()
        with self._lock:
            if not self._buf:
                return False
            return (cur - self._buf[-1][0]) <= max_age_s

    def latest_sample(self) -> Optional[tuple[float, float, float]]:
        """Trả mẫu mới nhất (t, yaw_unwrapped_rad, yawspeed_rad) hoặc None."""
        with self._lock:
            return self._buf[-1] if self._buf else None

    def yaw_at(self, t: float, max_extrap_s: float = 0.06) -> Optional[float]:
        """
        Nội suy tuyến tính góc yaw (rad, unwrapped) tại thời điểm t (monotonic).
        Cho phép ngoại suy tuyến tính bằng yawspeed tối đa `max_extrap_s` giây.
        Trả None nếu chưa có mẫu hoặc vượt quá giới hạn ngoại suy.
        """
        t = float(t)
        with self._lock:
            n = len(self._buf)
            if n == 0:
                return None

            t0, y0, w0 = self._buf[0]
            tn, yn, wn = self._buf[-1]

            if t <= t0:
                dt = t - t0
                if abs(dt) <= max_extrap_s:
                    return y0 + w0 * dt
                return None

            if t >= tn:
                dt = t - tn
                if dt <= max_extrap_s:
                    return yn + wn * dt
                return None

            # Tìm cặp mẫu kẹp t (thường nằm gần cuối bộ đệm nên duyệt ngược rất nhanh)
            for i in range(n - 2, -1, -1):
                ta, ya, _ = self._buf[i]
                tb, yb, _ = self._buf[i + 1]
                if ta <= t <= tb:
                    span = tb - ta
                    if span <= 1e-9:
                        return yb
                    alpha = (t - ta) / span
                    return ya + alpha * (yb - ya)
            return None

    def latest(self, now: Optional[float] = None, max_extrap_s: float = 0.08) -> Optional[float]:
        """Trả góc yaw ước lượng ở thời điểm hiện tại (có ngoại suy ngắn bằng yawspeed)."""
        cur = now if now is not None else time.monotonic()
        return self.yaw_at(cur, max_extrap_s=max_extrap_s)


class BearingProjector:
    """
    Chuyển đổi giữa tọa độ pixel ngang trên ảnh (0..W) và góc phương vị thế giới (rad)
    cũng như chiếu trụ thuận-nghịch bounding box cho ByteTrack.
    """
    def __init__(self, frame_width: int = 640, hfov_deg: float = 66.0):
        self.frame_width = float(frame_width)
        self.hfov_deg = float(hfov_deg)
        self.cx = self.frame_width / 2.0
        half_fov_rad = math.radians(max(1.0, min(175.0, self.hfov_deg)) / 2.0)
        self.focal_px = self.cx / math.tan(half_fov_rad)

    def px_to_bearing(self, x_px: float, psi_rad: float = 0.0) -> float:
        """Đổi tọa độ ngang x_px sang góc phương vị thế giới theta (rad)."""
        return float(psi_rad) + math.atan((float(x_px) - self.cx) / self.focal_px)

    def bearing_to_px(self, theta_rad: float, psi_rad: float = 0.0) -> float:
        """Chiếu góc phương vị thế giới theta (rad) về tọa độ ngang x_px trên ảnh."""
        rel = wrap_angle_rad(float(theta_rad) - float(psi_rad))
        rel = max(-1.45, min(1.45, rel))
        return self.cx + self.focal_px * math.tan(rel)

    def angle_rad_to_eq_px(self, angle_rad: float) -> float:
        """Quy đổi sai số góc (rad) sang pixel tương đương tuyến tính tại tâm ảnh."""
        return self.focal_px * float(angle_rad)

    def boxes_to_world(self, xyxy: np.ndarray, psi_rad: float) -> np.ndarray:
        """
        Chiếu tọa độ x1, x2 của mảng bbox (N, 4) sang không gian trụ thế giới
        để khử chuyển động xoay yaw của camera trước khi đưa vào ByteTrack.
        """
        arr = np.asarray(xyxy, dtype=np.float32).copy()
        if arr.size == 0:
            return arr
        f = self.focal_px
        cx = self.cx
        arr[:, 0] = cx + f * (psi_rad + np.arctan((arr[:, 0] - cx) / f))
        arr[:, 2] = cx + f * (psi_rad + np.arctan((arr[:, 2] - cx) / f))
        return arr

    def boxes_to_image(self, xyxy: np.ndarray, psi_rad: float) -> np.ndarray:
        """Chiếu ngược mảng bbox (N, 4) từ không gian trụ thế giới về tọa độ ảnh gốc."""
        arr = np.asarray(xyxy, dtype=np.float32).copy()
        if arr.size == 0:
            return arr
        f = self.focal_px
        cx = self.cx
        rel_x1 = np.clip((arr[:, 0] - cx) / f - psi_rad, -1.45, 1.45)
        rel_x2 = np.clip((arr[:, 2] - cx) / f - psi_rad, -1.45, 1.45)
        arr[:, 0] = np.clip(cx + f * np.tan(rel_x1), 0.0, self.frame_width)
        arr[:, 2] = np.clip(cx + f * np.tan(rel_x2), 0.0, self.frame_width)
        return arr


if __name__ == "__main__":
    print("=== Kiểm thử độc lập: ego_motion.py ===")
    yh = YawHistory(keep_s=2.0)
    yh.add(1.0, math.radians(179.0), math.radians(20.0))
    yh.add(1.1, math.radians(-179.0), math.radians(20.0))
    y_mid = yh.yaw_at(1.05)
    assert y_mid is not None and abs(math.degrees(y_mid) - 180.0) < 1e-3
    print(f"[PASS] Unwrap & nội suy qua ±180°: {math.degrees(y_mid):.2f}°")

    proj = BearingProjector(frame_width=640, hfov_deg=66.0)
    boxes = np.array([[100.0, 50.0, 220.0, 380.0], [420.0, 80.0, 560.0, 420.0]], dtype=np.float32)
    psi = math.radians(37.5)
    world = proj.boxes_to_world(boxes, psi)
    back = proj.boxes_to_image(world, psi)
    err = float(np.max(np.abs(back - boxes)))
    assert err < 1e-2, f"Round-trip error too large: {err}"
    print(f"[PASS] Chiếu thuận-nghịch bbox sai số tối đa: {err:.5f} px")
    print("=== Module ego_motion.py hoạt động tốt! ===")
