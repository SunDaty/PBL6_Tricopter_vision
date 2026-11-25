# src/controller.py
from dataclasses import dataclass
import math
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
    Bộ điều khiển PID/PD có vùng chết mềm (Soft Deadband) và giới hạn bão hòa (Saturation).
    Dùng chung cho cả điều khiển góc hướng Yaw (deg/s) và khoảng cách Tiến/Lùi Vx (m/s).
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
        """Xóa trạng thái tích phân và đạo hàm khi mất mục tiêu hoặc đổi mục tiêu."""
        self._integral = 0.0
        self._prev_error = None
        self._prev_time = None

    def compute(self, error: float, now: Optional[float] = None) -> float:
        """Tính giá trị điều khiển từ sai số hiện tại (có trừ vùng chết mềm Soft Deadband)."""
        current_time = now if now is not None else time.monotonic()

        if abs(error) <= self.deadband:
            self._integral = 0.0
            self._prev_error = 0.0
            self._prev_time = current_time
            return 0.0

        # Sai số hiệu dụng sau khi trừ vùng chết mềm (giúp lệnh tăng mượt từ 0, không nhảy bậc thang)
        eff_error = math.copysign(abs(error) - self.deadband, error)

        dt = (
            current_time - self._prev_time
            if (self._prev_time is not None and current_time > self._prev_time)
            else 0.0
        )

        p_term = self.kp * eff_error

        if dt > 0.0 and self.ki != 0.0:
            self._integral += eff_error * dt
            self._integral = max(min(self._integral, self.integral_limit), -self.integral_limit)
        i_term = self.ki * self._integral

        if dt > 0.0 and self._prev_error is not None and self.kd != 0.0:
            derivative = (eff_error - self._prev_error) / dt
        else:
            derivative = 0.0
        d_term = self.kd * derivative

        self._prev_error = eff_error
        self._prev_time = current_time

        output = p_term + i_term + d_term
        return max(min(output, self.max_out), self.min_out)


class HysteresisDeadband:
    """
    Vùng chết có độ trễ (Hysteresis Deadband):
    - Đi vào vùng chết (trả 0) khi |error| <= enter.
    - Chỉ thoát khỏi vùng chết khi |error| > exit (exit >= enter).
    Giúp tránh lệnh bật/tắt liên tục khi sai số dao động nhỏ quanh biên vùng chết.
    """
    def __init__(self, enter: float, exit: Optional[float] = None):
        self.enter = max(0.0, float(enter))
        self.exit = max(self.enter, float(exit) if exit is not None else self.enter)
        self._in_deadband = False

    @property
    def in_deadband(self) -> bool:
        return self._in_deadband

    def reset(self):
        self._in_deadband = False

    def apply(self, error: float) -> float:
        ae = abs(error)
        if self._in_deadband:
            if ae <= self.exit:
                return 0.0
            self._in_deadband = False
        elif ae <= self.enter:
            self._in_deadband = True
            return 0.0
        return math.copysign(max(0.0, ae - self.enter), error)


class SlewRateLimiter:
    """
    Giới hạn tốc độ biến thiên của lệnh điều khiển (ví dụ deg/s^2 cho yaw_rate),
    giúp đầu ra tăng/giảm và đảo chiều mượt, không gây giật cơ cấu chấp hành.
    """
    def __init__(self, max_rate: float, initial_dt: float = 0.1):
        self.max_rate = max(0.0, float(max_rate))
        self.initial_dt = max(0.01, float(initial_dt))
        self._value = 0.0
        self._last_time: Optional[float] = None

    @property
    def value(self) -> float:
        return self._value

    def reset(self, value: float = 0.0):
        self._value = float(value)
        self._last_time = None

    def step(self, target: float, now: Optional[float] = None) -> float:
        current_time = now if now is not None else time.monotonic()
        target = float(target)
        if self.max_rate <= 0.0:
            self._value = target
            self._last_time = current_time
            return self._value

        if self._last_time is None or current_time <= self._last_time:
            dt = self.initial_dt
        else:
            dt = min(0.5, current_time - self._last_time)
        self._last_time = current_time

        max_step = self.max_rate * dt
        delta = target - self._value
        if abs(delta) > max_step:
            self._value += math.copysign(max_step, delta)
        else:
            self._value = target
        return self._value


if __name__ == "__main__":
    print("=== Kiểm thử độc lập Module 1: controller.py (PIDController) ===")
    pid = PIDController(kp=0.1, kd=0.0, deadband=15.0, output_limits=(-25.0, 25.0))

    # 1. Sai số nằm trong vùng chết (<= 15)
    out_db = pid.compute(10.0, now=0.0)
    assert out_db == 0.0, f"Expected 0.0 in deadband, got {out_db}"
    print(f"[PASS] Deadband test (error=10.0px -> output={out_db})")

    # 2. Sai số dương vừa phải (115px - 15px deadband = 100px * 0.1 = 10.0)
    out_pos = pid.compute(115.0, now=1.0)
    assert abs(out_pos - 10.0) < 1e-5, f"Expected 10.0, got {out_pos}"
    print(f"[PASS] Soft-Deadband Proportional test (error=+115.0px -> output={out_pos:+.2f})")

    # 3. Sai số vượt ngưỡng bão hòa
    out_sat = pid.compute(-500.0, now=2.0)
    assert out_sat == -25.0, f"Expected -25.0 saturation, got {out_sat}"
    print(f"[PASS] Saturation test (error=-500.0px -> output={out_sat:+.2f})")
    print("=== Module controller.py hoạt động tốt! ===")
