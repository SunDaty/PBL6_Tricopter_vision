# src/controller.py
from dataclasses import dataclass
import time
from typing import Optional


@dataclass
class ControlCommand:
    """Lệnh vận tốc điều khiển drone trong hệ tọa độ thân (Body NED)."""
    vx: float = 0.0        # Vận tốc tiến (+) / lùi (-) (m/s)
    vy: float = 0.0        # Vận tốc dịch phải (+) / trái (-) (m/s)
    vz: float = 0.0        # Vận tốc hạ (+) / nâng (-) độ cao theo trục Down (m/s)
    yaw_rate: float = 0.0  # Tốc độ góc quay phải (+) / trái (-) (deg/s)


class PIDController:
    """
    Bộ điều khiển PID tổng quát có vùng chết (deadband) và giới hạn đầu ra (saturation).
    Tái sử dụng được cho cả góc Yaw, tiến/lùi (vx), trái/phải (vy) và độ cao (vz).
    """
    def __init__(
        self,
        kp: float,
        ki: float = 0.0,
        kd: float = 0.0,
        deadband: float = 0.0,
        output_limits: tuple[float, float] = (-float("inf"), float("inf")),
        integral_limit: float = 100.0,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.deadband = deadband
        self.min_out, self.max_out = output_limits
        self.integral_limit = integral_limit

        self._integral = 0.0
        self._prev_error: Optional[float] = None
        self._prev_time: Optional[float] = None

    def reset(self):
        """Xoá trạng thái tích phân và đạo hàm khi mất mục tiêu."""
        self._integral = 0.0
        self._prev_error = None
        self._prev_time = None

    def compute(self, error: float, now: Optional[float] = None) -> float:
        """Tính giá trị điều khiển từ sai số hiện tại."""
        if abs(error) < self.deadband:
            self.reset()
            return 0.0

        current_time = now if now is not None else time.monotonic()
        dt = (
            current_time - self._prev_time
            if (self._prev_time is not None and current_time > self._prev_time)
            else 0.0
        )

        # Khâu P (Proportional)
        p_term = self.kp * error

        # Khâu I (Integral) có chống bão hòa (Anti-windup)
        if dt > 0.0 and self.ki != 0.0:
            self._integral += error * dt
            self._integral = max(min(self._integral, self.integral_limit), -self.integral_limit)
        i_term = self.ki * self._integral

        # Khâu D (Derivative) giảm vọt lố khi mục tiêu di chuyển nhanh
        if dt > 0.0 and self._prev_error is not None and self.kd != 0.0:
            derivative = (error - self._prev_error) / dt
        else:
            derivative = 0.0
        d_term = self.kd * derivative

        self._prev_error = error
        self._prev_time = current_time

        output = p_term + i_term + d_term
        return max(min(output, self.max_out), self.min_out)


if __name__ == "__main__":
    print("=== Kiểm thử độc lập Module 1: controller.py (PIDController) ===")
    pid = PIDController(kp=0.1, kd=0.01, deadband=15.0, output_limits=(-25.0, 25.0))

    # 1. Sai số nằm trong vùng chết (< 15)
    out_db = pid.compute(10.0, now=0.0)
    assert out_db == 0.0, f"Expected 0.0 in deadband, got {out_db}"
    print(f"[PASS] Deadband test (error=10.0px -> output={out_db})")

    # 2. Sai số dương vừa phải
    out_pos = pid.compute(100.0, now=1.0)
    assert abs(out_pos - 10.0) < 1e-5, f"Expected 10.0, got {out_pos}"
    print(f"[PASS] Proportional test (error=+100.0px -> output={out_pos:+.2f})")

    # 3. Sai số vượt ngưỡng bão hòa
    out_sat = pid.compute(-500.0, now=2.0)
    assert out_sat == -25.0, f"Expected -25.0 saturation, got {out_sat}"
    print(f"[PASS] Saturation test (error=-500.0px -> output={out_sat:+.2f})")
    print("=== Module controller.py hoạt động tốt! ===")
