# src/mavlink_client.py
import argparse
import math
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

    def send_heartbeat_if_needed(self):
        """Gửi heartbeat định kỳ (1 Hz) từ Raspberry Pi 5 / PC tới Pixhawk 6C."""
        if self.dry_run or self.master is None:
            return
        now = time.monotonic()
        if now - self._last_heartbeat >= 1.0:
            self.master.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0,
                0,
                0,
            )
            self._last_heartbeat = now

    def send_control_command(self, cmd: ControlCommand):
        """
        Gửi lệnh vận tốc thân (vx, vy, vz) và tốc độ góc (yaw_rate) tới ArduPilot (chế độ GUIDED).
        - vx (m/s): Tiến (+) / Lùi (-)
        - yaw_rate (deg/s): Quay phải CW (+) / Quay trái CCW (-)
        """
        if self.dry_run or self.master is None:
            return

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

    def send_yaw_rate(self, yaw_rate_deg_s: float):
        """Chỉ gửi tốc độ góc quay Yaw (giữ nguyên vx = vy = vz = 0)."""
        self.send_control_command(ControlCommand(yaw_rate=yaw_rate_deg_s))

    def stop_drone(self):
        """Gửi lệnh dừng khẩn cấp (Hover đứng yên: vx=0, vy=0, vz=0, yaw_rate=0)."""
        self.send_control_command(ControlCommand())

    def close(self):
        """Gửi lệnh dừng drone và đóng cổng kết nối MAVLink."""
        self.stop_drone()
        if self.master is not None:
            self.master.close()
            self.master = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiểm thử độc lập Module MAVLinkClient")
    parser.add_argument("--conn", default="udpin:0.0.0.0:14550", help="Chuỗi kết nối UDP hoặc UART (/dev/ttyAMA0)")
    parser.add_argument("--baud", type=int, default=921600, help="Baudrate cho kết nối UART Pixhawk 6C")
    parser.add_argument("--test-mock", action="store_true", help="Chạy kiểm thử logic ở chế độ dry-run không cần SITL")
    args = parser.parse_args()

    print("=== Kiểm thử độc lập Module 7: mavlink_client.py ===")
    client = MAVLinkClient(conn_str=args.conn, baud=args.baud, dry_run=args.test_mock)
    client.connect()
    client.send_heartbeat_if_needed()
    client.send_yaw_rate(10.0)
    client.send_control_command(ControlCommand(vx=0.5, yaw_rate=-8.0))
    client.close()
    print("[PASS] Gửi Heartbeat, lệnh Yaw Rate và lệnh ControlCommand thành công!")
    print("=== Module mavlink_client.py hoạt động tốt! ===")
