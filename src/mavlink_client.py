# src/mavlink_client.py
import argparse
import math
import threading
import time
from pymavlink import mavutil

from controller import ControlCommand

# Bitmask cho SET_POSITION_TARGET_LOCAL_NED:
# Bỏ qua vị trí (x, y, z), gia tốc (afx, afy, afz) và góc yaw tuyệt đối.
# Sử dụng vận tốc thân (vx, vy, vz) và tốc độ góc (yaw_rate).
TYPE_MASK_VEL_AND_YAW_RATE = 0b010111000111  # 1479


class MAVLinkClient:
    """
    Module giao tiếp MAVLink độc lập dùng chung cho cả:
    1. Mô phỏng ArduPilot SITL (UDP: 'udpin:0.0.0.0:14550').
    2. Thực tế Mạch bay Pixhawk 6C (UART từ Raspberry Pi 5: '/dev/ttyAMA0', baud=921600).
    """
    def __init__(
        self,
        conn_str: str = "udpin:0.0.0.0:14550",
        baud: int = 921600,
        heartbeat_timeout: float = 15.0,
        dry_run: bool = False,
    ):
        self.conn_str = conn_str
        self.baud = baud
        self.heartbeat_timeout = heartbeat_timeout
        self.dry_run = dry_run

        self._start_time = time.monotonic()
        self._last_heartbeat = 0.0
        self.master = None

        # Debug: callback(t_arrival, yaw_deg, yawspeed_deg_s, fc_time_ms) và ClockSync (metrics.py)
        self.on_attitude = None
        self.clock_sync = None
        self._reader = None
        self._stop_reader = threading.Event()
        self._sim_yaw = 0.0
        self._last_timesync = 0.0

    def _start_reader(self):
        """Luồng nền đọc ATTITUDE & TIMESYNC phục vụ đo đáp ứng/đồng bộ đồng hồ."""
        if self.on_attitude is None or self.master is None:
            return
        # Yêu cầu ATTITUDE 30 Hz (MAV_CMD_SET_MESSAGE_INTERVAL = 511, msg id ATTITUDE = 30)
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            511, 0, 30, int(1e6 / 30), 0, 0, 0, 0, 0,
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
        """Dry-run: giả lập đáp ứng bậc 1 của drone (trễ 80ms, hệ số 0.8) để test đo đáp ứng."""
        def _emit():
            yawspeed = 0.8 * yaw_rate_cmd
            self._sim_yaw += yawspeed * 0.05
            self.on_attitude(time.monotonic(), self._sim_yaw, yawspeed, None)
        threading.Timer(delay_s, _emit).start()

    def send_control_command(self, cmd: ControlCommand) -> float:
        """
        Gửi lệnh vận tốc thân (vx, vy, vz) và tốc độ góc (yaw_rate) tới ArduPilot (chế độ GUIDED).
        - vx (m/s): Tiến (+) / Lùi (-)
        - yaw_rate (deg/s): Quay phải CW (+) / Quay trái CCW (-)
        Trả về thời điểm gửi (time.monotonic()) để đo độ trễ.
        """
        if self.dry_run or self.master is None:
            if self.dry_run and self.on_attitude is not None:
                self._simulate_attitude(cmd.yaw_rate)
            return time.monotonic()

        time_boot_ms = int((time.monotonic() - self._start_time) * 1000) & 0xFFFFFFFF
        yaw_rate_rad_s = math.radians(cmd.yaw_rate)

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
        return time.monotonic()

    def stop_drone(self):
        """Gửi lệnh dừng Hover (vx=0, vy=0, vz=0, yaw_rate=0) lặp 3 lần để chống rớt gói tin."""
        zero_cmd = ControlCommand()
        for _ in range(3):
            self.send_control_command(zero_cmd)
            if not self.dry_run and self.master is not None:
                time.sleep(0.02)

    def close(self):
        """Gửi lệnh dừng drone và đóng cổng kết nối MAVLink."""
        self.stop_drone()
        self._stop_reader.set()
        if self._reader is not None:
            self._reader.join(timeout=0.5)
        if self.master is not None:
            self.master.close()
            self.master = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiểm thử độc lập Module MAVLinkClient")
    parser.add_argument("--conn", default="udpin:0.0.0.0:14550", help="Chuỗi kết nối UDP hoặc UART (/dev/ttyAMA0)")
    parser.add_argument("--baud", type=int, default=921600, help="Baudrate cho kết nối UART Pixhawk 6C")
    parser.add_argument("--test-mock", action="store_true", help="Chạy kiểm thử logic ở chế độ dry-run không cần SITL")
    args = parser.parse_args()

    print("=== Kiểm thử độc lập Module 3: mavlink_client.py ===")
    client = MAVLinkClient(conn_str=args.conn, baud=args.baud, dry_run=args.test_mock)
    client.connect()
    client.send_heartbeat_if_needed()
    client.send_control_command(ControlCommand(vx=0.5, yaw_rate=-8.0))
    client.close()
    print("[PASS] Gửi Heartbeat, lệnh ControlCommand và dừng drone thành công!")
    print("=== Module mavlink_client.py hoạt động tốt! ===")
