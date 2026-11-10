# src/yaw_tracker.py
from typing import Optional
from controller import PIDController


class YawOnlyTracker:
    """
    Module điều khiển ĐỘC LẬP chỉ dành riêng cho góc hướng (Yaw).
    - Nhận vào: Bounding Box (x1, y1, x2, y2) của mục tiêu.
    - Tính toán: Sai số ngang error_x (pixel) so với tâm khung hình.
    - Xuất ra: Tốc độ góc quay yaw_rate (deg/s).
      + yaw_rate > 0: Mục tiêu bên phải -> Quay phải (CW).
      + yaw_rate < 0: Mục tiêu bên trái -> Quay trái (CCW).
      + yaw_rate = 0: Mục tiêu nằm trong vùng chết (deadband) hoặc không có mục tiêu.
    """
    def __init__(
        self,
        frame_width: int = 640,
        yaw_kp: float = 0.08,
        yaw_kd: float = 0.01,
        yaw_deadband: float = 15.0,
        max_yaw_rate: float = 25.0,
    ):
        self.frame_width = frame_width
        self.center_x = frame_width / 2.0
        self.yaw_deadband = yaw_deadband
        self.max_yaw_rate = max_yaw_rate

        self.pid = PIDController(
            kp=yaw_kp,
            kd=yaw_kd,
            deadband=yaw_deadband,
            output_limits=(-max_yaw_rate, max_yaw_rate),
        )

    def reset(self):
        """Xóa trạng thái bộ điều khiển PD góc Yaw."""
        self.pid.reset()

    def get_error_x(self, bbox: Optional[tuple[int, int, int, int]]) -> Optional[float]:
        """Tính độ lệch tâm ngang (pixel) từ tâm bbox tới tâm màn hình."""
        if bbox is None:
            return None
        x1, _, x2, _ = bbox
        target_x = (x1 + x2) / 2.0
        return target_x - self.center_x

    def compute_yaw_rate(
        self,
        bbox: Optional[tuple[int, int, int, int]],
        now: Optional[float] = None,
    ) -> float:
        """Tính tốc độ góc quay Yaw (deg/s) từ bounding box."""
        error_x = self.get_error_x(bbox)
        if error_x is None:
            self.reset()
            return 0.0
        return self.pid.compute(error_x, now=now)


if __name__ == "__main__":
    print("=== Kiểm thử độc lập Module 3: yaw_tracker.py (YawOnlyTracker) ===")
    yaw_ctrl = YawOnlyTracker(frame_width=640, yaw_kp=0.08, yaw_deadband=15.0)

    # 1. Mục tiêu ở bên phải tâm màn hình (tâm x = 440 -> lệch +120px)
    rate_right = yaw_ctrl.compute_yaw_rate((400, 100, 480, 360), now=0.0)
    assert rate_right > 0, f"Expected positive yaw_rate for right target, got {rate_right}"
    print(f"[PASS] Mục tiêu lệch phải (+120px) -> yaw_rate = {rate_right:+.2f} deg/s (Quay phải)")

    # 2. Mục tiêu ở bên trái tâm màn hình (tâm x = 200 -> lệch -120px)
    yaw_ctrl.reset()
    rate_left = yaw_ctrl.compute_yaw_rate((160, 100, 240, 360), now=1.0)
    assert rate_left < 0, f"Expected negative yaw_rate for left target, got {rate_left}"
    print(f"[PASS] Mục tiêu lệch trái (-120px) -> yaw_rate = {rate_left:+.2f} deg/s (Quay trái)")

    # 3. Mục tiêu ở chính giữa tâm màn hình (tâm x = 325 -> lệch +5px < deadband 15px)
    rate_center = yaw_ctrl.compute_yaw_rate((285, 100, 365, 360), now=2.0)
    assert rate_center == 0.0, f"Expected 0.0 in deadband, got {rate_center}"
    print(f"[PASS] Mục tiêu trong Deadband (+5px) -> yaw_rate = {rate_center:+.2f} deg/s (Đứng yên)")
    print("=== Module yaw_tracker.py hoạt động tốt! ===")
