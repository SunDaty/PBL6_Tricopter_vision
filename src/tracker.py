# src/tracker.py
from collections import deque
import math
import time
from typing import Optional

from controller import ControlCommand, PIDController
from detector import DetectedPerson


class TargetTracker:
    """
    Module điều khiển bám mục tiêu hợp nhất (dùng chung cho Gazebo Sim & Raspberry Pi 5):
    1. Chọn và khóa ID mục tiêu khi người dùng Click chuột trái (`select_by_point`).
    2. Điều khiển góc quay Yaw (`yaw_rate` deg/s) giữ mục tiêu ở tâm ngang khung hình.
    3. Điều khiển khoảng cách Tiến/Lùi (`vx` m/s) độ trễ thấp & mượt:
       - Lọc Median 3-frame + EMA nhanh (alpha=0.55) giúp khử nhiễu nhảy bbox mà không gây trễ pha.
       - Hàm sai số khoảng cách đối xứng chuẩn quang học:
           + Đi xa (cur_h < ref_h)  : e = (ref_h / cur_h) - 1.0  (tỉ lệ thuận với khoảng cách thực Z)
           + Lại gần (cur_h > ref_h): e = -((cur_h / ref_h) - 1.0)
       - Vùng chết nhỏ ±5% kèm vận tốc khởi động tối thiểu (min_vx=0.15 m/s) để thắng quán tính ngay.
       - Bộ giới hạn gia tốc (Slew-Rate Limiter) giúp mũi Tricopter nghiêng Pitch êm ái, không giật cục.
    """
    def __init__(
        self,
        frame_width: int = 640,
        frame_height: int = 480,
        yaw_kp: float = 0.08,
        yaw_kd: float = 0.01,
        yaw_deadband_px: float = 15.0,
        max_yaw_rate: float = 25.0,
        dist_kp: float = 1.6,
        dist_deadband_ratio: float = 0.05,
        min_vx: float = 0.15,
        max_vx: float = 1.2,
        max_vx_accel: float = 1.8,
        max_vx_decel: float = 2.5,
        height_ema_alpha: float = 0.55,
        lost_timeout: float = 2.0,
        auto_lock: bool = False,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.center_x = frame_width / 2.0
        self.yaw_deadband_px = yaw_deadband_px

        self.dist_kp = dist_kp
        self.dist_deadband_ratio = dist_deadband_ratio
        self.min_vx = min_vx
        self.max_vx = max_vx
        self.max_vx_accel = max_vx_accel
        self.max_vx_decel = max_vx_decel
        self.height_ema_alpha = height_ema_alpha
        self.lost_timeout = lost_timeout
        self.auto_lock = auto_lock

        # Trạng thái mục tiêu đang khóa
        self.locked_id: Optional[int] = None
        self.last_seen_time: Optional[float] = None
        self.ref_height_px: Optional[float] = None
        self._ref_width_px: Optional[float] = None
        self.current_height_px: Optional[float] = None
        self._height_window: deque[float] = deque(maxlen=3)
        self._current_vx: float = 0.0
        self._last_vx_time: Optional[float] = None

        # Bộ điều khiển PD cho góc Yaw (đầu vào: error_x_px, đầu ra: deg/s)
        self.yaw_pid = PIDController(
            kp=yaw_kp,
            kd=yaw_kd,
            deadband=yaw_deadband_px,
            output_limits=(-max_yaw_rate, max_yaw_rate),
        )

    def reset(self):
        """Xóa mục tiêu đang khóa và đưa bộ điều khiển về trạng thái chờ (Hover)."""
        self.locked_id = None
        self.last_seen_time = None
        self.ref_height_px = None
        self._ref_width_px = None
        self.current_height_px = None
        self._height_window.clear()
        self._current_vx = 0.0
        self._last_vx_time = None
        self.yaw_pid.reset()

    def _calibrate_reference(self, person: DetectedPerson, now: float):
        """Chốt chiều cao và bề rộng khung bao lúc vừa chọn mục tiêu làm mốc chuẩn."""
        raw_h = max(50.0, min(float(self.frame_height * 0.85), person.height))
        self.ref_height_px = raw_h
        self._ref_width_px = max(20.0, person.width)
        self.current_height_px = raw_h
        self._height_window.clear()
        self._height_window.append(raw_h)
        self._current_vx = 0.0
        self._last_vx_time = now
        self.yaw_pid.reset()

    def select_by_point(
        self,
        x: int,
        y: int,
        detections: list[DetectedPerson],
        now: Optional[float] = None,
    ) -> Optional[int]:
        """
        Chọn mục tiêu tại tọa độ click chuột trái (x, y) và chốt ngay cự ly hiện tại làm chuẩn.
        Nếu nhiều khung bao chồng lấn, ưu tiên người có tâm gần điểm click nhất.
        """
        candidates = [d for d in detections if d.contains_point(x, y)]
        if not candidates:
            return None

        current_time = now if now is not None else time.monotonic()
        chosen = min(
            candidates,
            key=lambda d: (d.center[0] - x) ** 2 + (d.center[1] - y) ** 2,
        )
        self.locked_id = chosen.track_id
        self.last_seen_time = current_time
        self._calibrate_reference(chosen, now=current_time)
        return chosen.track_id

    def get_occlusion_elapsed(self, now: Optional[float] = None) -> float:
        """Thời gian (giây) kể từ lần cuối nhìn thấy mục tiêu đang khóa."""
        if self.locked_id is None or self.last_seen_time is None:
            return 0.0
        current_time = now if now is not None else time.monotonic()
        return max(0.0, current_time - self.last_seen_time)

    def _compute_effective_height(self, person: DetectedPerson) -> float:
        """
        Tính chiều cao mục tiêu có khử nhiễu gai (Median-of-3) + EMA nhanh (alpha=0.55):
        - Khi người tiến quá sát làm chân/đầu chạm mép ảnh (y2 >= H-8 hoặc y1 <= 8)
          và bề rộng phình to hơn lúc khóa, ước lượng qua tỉ lệ bề rộng để drone lùi lại an toàn.
        """
        _, y1, _, y2 = person.bbox
        raw_h = person.height
        raw_w = person.width

        is_clipped = (y2 >= self.frame_height - 8) or (y1 <= 8)
        if is_clipped and self.ref_height_px is not None and self._ref_width_px is not None:
            width_scale = raw_w / self._ref_width_px
            if width_scale > 1.08:
                raw_h = max(raw_h, self.ref_height_px * width_scale)

        self._height_window.append(raw_h)
        sorted_w = sorted(self._height_window)
        median_h = sorted_w[len(sorted_w) // 2]

        if self.current_height_px is None:
            self.current_height_px = median_h
        else:
            self.current_height_px = (
                self.height_ema_alpha * median_h
                + (1.0 - self.height_ema_alpha) * self.current_height_px
            )
        return self.current_height_px

    def _compute_smooth_vx(self, cur_h: float, now: float) -> float:
        """
        Tính vận tốc tiến/lùi vx (m/s) có bù quán tính khởi động và giới hạn gia tốc (Slew-Rate).
        """
        if self.ref_height_px is None or cur_h <= 1.0:
            self._current_vx = 0.0
            self._last_vx_time = now
            return 0.0

        # Sai số khoảng cách đối xứng chuẩn quang học:
        # > 0 khi người đi xa (cur_h < ref_h), < 0 khi người lại gần (cur_h > ref_h)
        if cur_h <= self.ref_height_px:
            dist_error = (self.ref_height_px / cur_h) - 1.0
        else:
            dist_error = -((cur_h / self.ref_height_px) - 1.0)

        # Vùng chết mềm ±5% kèm bù vận tốc khởi động tối thiểu (min_vx)
        if abs(dist_error) <= self.dist_deadband_ratio:
            target_vx = 0.0
        else:
            eff_err = math.copysign(abs(dist_error) - self.dist_deadband_ratio, dist_error)
            raw_vx = math.copysign(self.min_vx, eff_err) + self.dist_kp * eff_err
            target_vx = max(-self.max_vx, min(self.max_vx, raw_vx))

        # Giới hạn gia tốc (Slew-Rate Limiter) giúp lệnh vận tốc biến thiên mượt mà
        dt = (
            now - self._last_vx_time
            if (self._last_vx_time is not None and now > self._last_vx_time)
            else 0.0
        )
        self._last_vx_time = now

        if dt <= 0.0 or dt > 0.5:
            # Nếu là bước đầu tiên hoặc sau quãng nghỉ dài, cho phép cập nhật trực tiếp
            self._current_vx = target_vx
            return self._current_vx

        delta_vx = target_vx - self._current_vx
        # Dùng max_vx_decel khi đang giảm tốc về 0 hoặc đảo chiều, dùng max_vx_accel khi tăng tốc
        is_braking = (abs(target_vx) < abs(self._current_vx)) or (target_vx * self._current_vx < 0.0)
        rate_limit = self.max_vx_decel if is_braking else self.max_vx_accel
        max_step = rate_limit * dt

        if abs(delta_vx) > max_step:
            self._current_vx += math.copysign(max_step, delta_vx)
        else:
            self._current_vx = target_vx

        return self._current_vx

    def update(
        self,
        detections: list[DetectedPerson],
        now: Optional[float] = None,
    ) -> tuple[ControlCommand, Optional[DetectedPerson]]:
        """
        Cập nhật danh sách phát hiện ở frame hiện tại và tính lệnh điều khiển (vx, yaw_rate).
        - Nếu chưa chọn mục tiêu hoặc đang khuất bóng: trả về lệnh 0 (Hover).
        - Nếu mất dấu quá `lost_timeout`: tự động hủy khóa về trạng thái chờ chọn lại.
        """
        current_time = now if now is not None else time.monotonic()

        if self.locked_id is None:
            if self.auto_lock and detections:
                chosen = min(detections, key=lambda d: d.track_id)
                self.locked_id = chosen.track_id
                self.last_seen_time = current_time
                self._calibrate_reference(chosen, now=current_time)
            else:
                return ControlCommand(), None

        target = next((d for d in detections if d.track_id == self.locked_id), None)
        if target is None:
            # Mục tiêu tạm thời khuất bóng -> giảm tốc mượt hoặc Hover
            self.yaw_pid.reset()
            self._current_vx = 0.0
            self._last_vx_time = current_time
            if self.get_occlusion_elapsed(now=current_time) > self.lost_timeout:
                self.reset()
            return ControlCommand(), None

        self.last_seen_time = current_time
        if self.ref_height_px is None:
            self._calibrate_reference(target, now=current_time)

        # 1. Tính tốc độ góc quay Yaw (deg/s) từ độ lệch ngang so với tâm màn hình
        error_x = target.center[0] - self.center_x
        yaw_rate = self.yaw_pid.compute(error_x, now=current_time)

        # 2. Tính vận tốc Tiến/Lùi Vx (m/s) mượt và độ trễ thấp
        cur_h = self._compute_effective_height(target)
        vx = self._compute_smooth_vx(cur_h, now=current_time)

        return ControlCommand(vx=vx, vy=0.0, vz=0.0, yaw_rate=yaw_rate), target


# Alias tương thích
CombinedTracker = TargetTracker


if __name__ == "__main__":
    print("=== Kiểm thử độc lập Module 5: tracker.py (TargetTracker: Select + Yaw + Smooth Distance) ===")
    tracker = TargetTracker(height_ema_alpha=1.0, lost_timeout=2.0)

    # 1. Chưa chọn mục tiêu -> Drone Hover đứng yên (vx=0, yaw_rate=0)
    p1 = DetectedPerson(track_id=1, bbox=(400, 170, 480, 310), conf=0.92)
    cmd_idle, active = tracker.update([p1], now=0.0)
    assert active is None and cmd_idle.vx == 0.0 and cmd_idle.yaw_rate == 0.0
    print("[PASS] 1. Khi chưa click chọn mục tiêu -> Drone Hover đứng yên")

    # 2. Click chuột trái vào người #1 (cao 140px, tâm x=440 lệch phải +120px)
    chosen_id = tracker.select_by_point(440, 240, [p1], now=0.1)
    cmd_lock, active = tracker.update([p1], now=0.1)
    assert chosen_id == 1 and active is not None
    assert cmd_lock.vx == 0.0 and cmd_lock.yaw_rate > 0.0
    print(
        f"[PASS] 2. Click khóa ID #{chosen_id} (ref_h={tracker.ref_height_px:.0f}px): "
        f"yaw_rate={cmd_lock.yaw_rate:+.2f} deg/s, vx={cmd_lock.vx:+.2f} m/s (Không bị tự trôi)"
    )

    # 3. Người đi ra xa (chiều cao giảm xuống 90px) -> Tăng tốc mượt tiến theo (vx > 0)
    p1_far = DetectedPerson(track_id=1, bbox=(400, 195, 480, 285), conf=0.92)
    tracker.update([p1_far], now=0.3)
    cmd_far, _ = tracker.update([p1_far], now=0.5)
    assert cmd_far.vx > 0.0
    print(f"[PASS] 3. Mục tiêu đi xa (h={tracker.current_height_px:.0f}px) -> vx={cmd_far.vx:+.2f} m/s (Tiến theo mượt)")

    # 4. Người bước lại gần (chiều cao tăng lên 200px) -> Đảo chiều mượt lùi lại (vx < 0)
    p1_close = DetectedPerson(track_id=1, bbox=(380, 140, 500, 340), conf=0.95)
    tracker.update([p1_close], now=0.8)
    cmd_close, _ = tracker.update([p1_close], now=1.2)
    assert cmd_close.vx < 0.0
    print(f"[PASS] 4. Mục tiêu lại gần (h={tracker.current_height_px:.0f}px) -> vx={cmd_close.vx:+.2f} m/s (Lùi lại mượt)")

    # 5. Mất dấu quá lost_timeout (2.0s) -> Tự động về trạng thái chờ chọn lại
    tracker.update([], now=3.5)
    assert tracker.locked_id is None
    print("[PASS] 5. Khuất bóng quá 2.0s -> Tự động hủy khóa và Hover an toàn")
    print("=== Module tracker.py hoạt động tốt! ===")
