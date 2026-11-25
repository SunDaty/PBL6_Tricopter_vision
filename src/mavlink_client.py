# src/mavlink_client.py
import argparse
import math
import threading
import time
from pymavlink import mavutil

from controller import ControlCommand
from ego_motion import YawHistory

# Bitmask cho SET_POSITION_TARGET_LOCAL_NED:
# Bỏ qua vị trí (x, y, z), gia tốc (afx, afy, afz) và góc yaw tuyệt đối.
# Sử dụng vận tốc thân (vx, vy, vz) và tốc độ góc (yaw_rate).
TYPE_MASK_VEL_AND_YAW_RATE = 0b010111000111  # 1479

# Bitmask cho SET_ATTITUDE_TARGET (GUIDED_NOGPS):
# Bỏ qua body_roll_rate (1) và body_pitch_rate (2), giữ attitude [1,0,0,0] và body_yaw_rate.
TYPE_MASK_ATTITUDE_YAW_RATE = 0b00000011

VALID_CONTROL_METHODS = ("guided_vel", "guided_nogps", "rc_override")


class MAVLinkClient:
    """
    Module giao tiếp MAVLink độc lập dùng chung cho cả:
    1. Mô phỏng ArduPilot SITL (UDP: 'udpin:0.0.0.0:14550').
    2. Thực tế Mạch bay Pixhawk 6C (UART từ Raspberry Pi 5: '/dev/ttyAMA0', baud=921600).
    Hỗ trợ 3 phương thức điều khiển:
    - 'guided_vel'  : SET_POSITION_TARGET_LOCAL_NED (vx + yaw_rate)
    - 'guided_nogps': SET_ATTITUDE_TARGET (body_yaw_rate + thrust)
    - 'rc_override' : RC_CHANNELS_OVERRIDE kênh 4 (Yaw / Tail Servo PWM)
    """
    def __init__(
        self,
        conn_str: str = "/dev/ttyAMA0",
        baud: int = 921600,
        heartbeat_timeout: float = 15.0,
        dry_run: bool = False,
        control_method: str = "guided_vel",
        max_yaw_rate: float = 25.0,
        nogps_thrust: float = 0.0,
        rc_yaw_center: int = 1500,
        rc_yaw_range: int = 150,
        telemetry_enabled: bool = True,
        attitude_hz: int = 50,
    ):
        method = control_method.lower()
        if method not in VALID_CONTROL_METHODS:
            raise ValueError(
                f"control_method '{control_method}' không hợp lệ. Chọn một trong: {VALID_CONTROL_METHODS}"
            )
        self.conn_str = conn_str
        self.baud = baud
        self.heartbeat_timeout = heartbeat_timeout
        self.dry_run = dry_run
        self.control_method = method
        self.max_yaw_rate = max(1.0, float(max_yaw_rate))
        self.nogps_thrust = max(0.0, min(1.0, float(nogps_thrust)))
        self.rc_yaw_center = int(rc_yaw_center)
        self.rc_yaw_range = int(rc_yaw_range)
        self.telemetry_enabled = bool(telemetry_enabled)
        self.attitude_hz = max(1, int(attitude_hz))

        self._start_time = time.monotonic()
        self._last_heartbeat = 0.0
        self.master = None

        # Lịch sử góc Yaw (rad, unwrapped) phục vụ bù chuyển động & bù độ trễ
        self.yaw_history = YawHistory(keep_s=3.0)

        # Debug: callback(t_arrival, yaw_deg, yawspeed_deg_s, fc_time_ms) và ClockSync (metrics.py)
        self.on_attitude = None
        self.clock_sync = None
        self._reader = None
        self._stop_reader = threading.Event()
        self._sim_yaw = 0.0
        self._last_timesync = 0.0

    def _start_reader(self):
        """Luồng nền đọc ATTITUDE (50 Hz mặc định) & TIMESYNC phục vụ điều khiển bù trễ và đo đáp ứng."""
        if self.master is None or (not self.telemetry_enabled and self.on_attitude is None):
            return
        # Yêu cầu ATTITUDE theo attitude_hz (MAV_CMD_SET_MESSAGE_INTERVAL = 511, msg id ATTITUDE = 30)
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            511, 0, 30, int(1e6 / self.attitude_hz), 0, 0, 0, 0, 0,
        )
        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()

    def _reader_loop(self):
        while not self._stop_reader.is_set() and self.master is not None:
            try:
                msg = self.master.recv_match(type=["ATTITUDE", "TIMESYNC"], blocking=True, timeout=0.1)
            except Exception:
                continue
            if msg is None:
                continue
            now = time.monotonic()
            if msg.get_type() == "ATTITUDE":
                t_sample = now
                if self.clock_sync is not None:
                    conv = self.clock_sync.to_pi_time(float(msg.time_boot_ms) * 1e6)
                    if conv is not None and abs(conv - now) < 0.25:
                        t_sample = conv
                self.yaw_history.add(t_sample, float(msg.yaw), float(msg.yawspeed))
                if self.on_attitude is not None:
                    self.on_attitude(
                        now, math.degrees(msg.yaw), math.degrees(msg.yawspeed), float(msg.time_boot_ms)
                    )
            elif msg.get_type() == "TIMESYNC" and self.clock_sync is not None:
                if msg.tc1 != 0 and msg.ts1 != 0:  # phản hồi cho yêu cầu của mình
                    self.clock_sync.update(msg.ts1, msg.tc1, time.monotonic_ns())

    def connect(self):
        """Thiết lập kết nối MAVLink và đợi Heartbeat từ ArduPilot / Pixhawk 6C."""
        if self.dry_run:
            print(f"[*] [MAVLinkClient - DRY RUN] Giả lập kết nối tới {self.conn_str} (baud={self.baud})")
            return

        print(f"[*] [MAVLinkClient] Đang kết nối MAVLink tới {self.conn_str} (baud={self.baud})...")
        self.master = mavutil.mavlink_connection(
            self.conn_str,
            baud=self.baud,
            source_system=255,
            source_component=mavutil.mavlink.MAV_COMP_ID_ONBOARD_COMPUTER,
        )
        heartbeat = self.master.wait_heartbeat(timeout=self.heartbeat_timeout)
        if heartbeat is None:
            self.master.close()
            raise TimeoutError(
                f"Không nhận được heartbeat MAVLink từ {self.conn_str} trong {self.heartbeat_timeout}s. "
                "Kiểm tra lại ArduPilot SITL hoặc dây cáp UART nối Pixhawk 6C (TELEM2)."
            )
        print(
            f"[*] [MAVLinkClient] Đã kết nối thành công "
            f"(system={self.master.target_system}, component={self.master.target_component})!"
        )
        self._start_reader()

    def send_heartbeat_if_needed(self):
        """Gửi heartbeat định kỳ (1 Hz) từ Raspberry Pi 5 / PC tới Pixhawk 6C."""
        if self.dry_run or self.master is None:
            return
        now = time.monotonic()
        if self.clock_sync is not None and now - self._last_timesync >= 1.0:
            self.master.mav.timesync_send(0, time.monotonic_ns())  # tc1=0: yêu cầu, ts1=ns của Pi
            self._last_timesync = now
        if now - self._last_heartbeat >= 1.0:
            self.master.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0,
                0,
                0,
            )
            self._last_heartbeat = now

    def _simulate_attitude(self, yaw_rate_cmd: float, delay_s: float = 0.08):
        """Dry-run: giả lập đáp ứng bậc 1 của drone (trễ 80ms, hệ số 0.8) để test bù trễ/đo đáp ứng."""
        def _emit():
            if self._stop_reader.is_set():
                return
            t_now = time.monotonic()
            yawspeed = 0.8 * yaw_rate_cmd
            self._sim_yaw += yawspeed * 0.05
            self.yaw_history.add(t_now, math.radians(self._sim_yaw), math.radians(yawspeed))
            if self.on_attitude is not None:
                self.on_attitude(t_now, self._sim_yaw, yawspeed, None)
        timer = threading.Timer(delay_s, _emit)
        timer.daemon = True
        timer.start()

    def yaw_rate_to_pwm(self, yaw_rate_deg_s: float) -> int:
        """Quy đổi tốc độ góc Yaw (deg/s) sang xung PWM kênh 4 quanh giá trị trung tâm (1500 us)."""
        norm = max(-1.0, min(1.0, float(yaw_rate_deg_s) / self.max_yaw_rate))
        return int(round(self.rc_yaw_center + norm * self.rc_yaw_range))

    def release_rc_override(self):
        """Trả quyền điều khiển các kênh RC về tay cầm phát sóng."""
        if self.dry_run or self.master is None:
            return
        self.master.mav.rc_channels_override_send(
            self.master.target_system,
            self.master.target_component,
            0, 0, 0, 0, 0, 0, 0, 0,
        )

    def send_control_command(self, cmd: ControlCommand) -> float:
        """
        Gửi lệnh điều khiển tới ArduPilot / Pixhawk 6C theo phương thức đã chọn.
        Trả về thời điểm gửi (time.monotonic()) để đo độ trễ.
        """
        if self.dry_run or self.master is None:
            if self.dry_run and (self.telemetry_enabled or self.on_attitude is not None):
                self._simulate_attitude(cmd.yaw_rate)
            return time.monotonic()

        time_boot_ms = int((time.monotonic() - self._start_time) * 1000) & 0xFFFFFFFF
        yaw_rate_rad_s = math.radians(cmd.yaw_rate)

        if self.control_method == "guided_vel":
            self.master.mav.set_position_target_local_ned_send(
                time_boot_ms,
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_FRAME_BODY_NED,
                TYPE_MASK_VEL_AND_YAW_RATE,
                0.0, 0.0, 0.0,           # x, y, z (bỏ qua)
                cmd.vx, cmd.vy, cmd.vz,  # vx, vy, vz (m/s - hệ tọa độ thân drone)
                0.0, 0.0, 0.0,           # afx, afy, afz (bỏ qua)
                0.0,                     # yaw (bỏ qua)
                yaw_rate_rad_s,          # yaw_rate (rad/s)
            )
        elif self.control_method == "guided_nogps":
            self.master.mav.set_attitude_target_send(
                time_boot_ms,
                self.master.target_system,
                self.master.target_component,
                TYPE_MASK_ATTITUDE_YAW_RATE,
                [1.0, 0.0, 0.0, 0.0],
                0.0,
                0.0,
                yaw_rate_rad_s,
                self.nogps_thrust,
            )
        elif self.control_method == "rc_override":
            yaw_pwm = self.yaw_rate_to_pwm(cmd.yaw_rate)
            self.master.mav.rc_channels_override_send(
                self.master.target_system,
                self.master.target_component,
                65535, 65535, 65535, yaw_pwm,
                65535, 65535, 65535, 65535,
            )
        return time.monotonic()

    def stop_drone(self):
        """Gửi lệnh dừng Hover (vx=0, vy=0, vz=0, yaw_rate=0) lặp 3 lần để chống rớt gói tin."""
        zero_cmd = ControlCommand()
        for _ in range(3):
            self.send_control_command(zero_cmd)
            if not self.dry_run and self.master is not None:
                time.sleep(0.02)

    def close(self):
        """Gửi lệnh dừng drone, giải phóng RC override (nếu có) và đóng cổng MAVLink."""
        self.stop_drone()
        if self.control_method == "rc_override":
            self.release_rc_override()
        self._stop_reader.set()
        if self._reader is not None:
            self._reader.join(timeout=0.5)
        if self.master is not None:
            self.master.close()
            self.master = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiểm thử độc lập Module MAVLinkClient")
    parser.add_argument("--conn", default="/dev/ttyAMA0", help="Chuỗi kết nối UART (/dev/ttyAMA0) hoặc UDP (udpin:0.0.0.0:14550)")
    parser.add_argument("--baud", type=int, default=921600, help="Baudrate cho kết nối UART Pixhawk 6C")
    parser.add_argument("--method", choices=list(VALID_CONTROL_METHODS), default="guided_vel", help="Phương thức điều khiển MAVLink")
    parser.add_argument("--test-mock", action="store_true", help="Chạy kiểm thử logic ở chế độ dry-run không cần Pixhawk thật")
    args = parser.parse_args()

    print("=== Kiểm thử độc lập Module 3: mavlink_client.py ===")
    client = MAVLinkClient(conn_str=args.conn, baud=args.baud, dry_run=args.test_mock, control_method=args.method)
    assert client.yaw_rate_to_pwm(0.0) == 1500
    assert client.yaw_rate_to_pwm(25.0) == 1650
    assert client.yaw_rate_to_pwm(-25.0) == 1350
    client.connect()
    client.send_heartbeat_if_needed()
    client.send_control_command(ControlCommand(vx=0.5, yaw_rate=-8.0))
    client.close()
    print("[PASS] Gửi Heartbeat, lệnh ControlCommand và dừng drone thành công!")
    print("=== Module mavlink_client.py hoạt động tốt! ===")
