#!/usr/bin/env python3
"""
Công cụ CHỈ ĐỌC (read-only) các tham số điều khiển trục Yaw & Servo đuôi trên Pixhawk 6C / ArduCopter.
KHÔNG thay đổi hay ghi bất kỳ tham số nào xuống mạch bay.

Cách dùng:
    /home/dat/drone_tracking/drone_env/bin/python tools/read_fc_yaw_params.py
    /home/dat/drone_tracking/drone_env/bin/python tools/read_fc_yaw_params.py --conn /dev/ttyAMA0 --baud 921600
"""
import argparse
import time
from pymavlink import mavutil

YAW_PARAMS = [
    "FRAME_CLASS",
    "FRAME_TYPE",
    "ATC_ANG_YAW_P",
    "ATC_RAT_YAW_P",
    "ATC_RAT_YAW_I",
    "ATC_RAT_YAW_D",
    "ATC_RAT_YAW_FF",
    "ATC_RAT_YAW_IMAX",
    "ATC_RAT_YAW_FLTT",
    "ATC_RAT_YAW_FLTE",
    "ATC_RAT_YAW_FLTD",
    "ATC_ACCEL_Y_MAX",
    "ATC_SLEW_YAW",
    "ATC_RATE_Y_MAX",
    "PILOT_Y_RATE",
    "MOT_YAW_SV_ANGLE",
    "SERVO7_FUNCTION",
    "SERVO7_MIN",
    "SERVO7_TRIM",
    "SERVO7_MAX",
    "SERVO7_REVERSED",
    "GUID_OPTIONS",
]


def read_param(master, name: str, timeout: float = 0.6):
    master.mav.param_request_read_send(
        master.target_system,
        master.target_component,
        name.encode("ascii"),
        -1,
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        msg = master.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.1)
        if msg is None:
            continue
        pid = msg.param_id.decode("ascii", errors="ignore").rstrip("\x00") if isinstance(msg.param_id, bytes) else str(msg.param_id).rstrip("\x00")
        if pid == name:
            return msg.param_value
    return None


def main():
    parser = argparse.ArgumentParser(description="Đọc (read-only) tham số trục Yaw của ArduCopter trên Pixhawk 6C.")
    parser.add_argument("--conn", default="/dev/ttyAMA0", help="Chuỗi kết nối MAVLink (mặc định: /dev/ttyAMA0)")
    parser.add_argument("--baud", type=int, default=921600, help="Baudrate UART (mặc định: 921600)")
    parser.add_argument("--timeout", type=float, default=10.0, help="Thời gian chờ heartbeat (giây)")
    args = parser.parse_args()

    print(f"[*] Kết nối read-only tới {args.conn} @ {args.baud}...")
    master = mavutil.mavlink_connection(args.conn, baud=args.baud)
    hb = master.wait_heartbeat(timeout=args.timeout)
    if hb is None:
        master.close()
        raise SystemExit(f"[!] Không nhận được heartbeat từ {args.conn} trong {args.timeout}s.")

    print(f"[*] Đã kết nối (system={master.target_system}, component={master.target_component}). Đang đọc tham số Yaw...\n")
    print(f"{'THAM SỐ':<22} | {'GIÁ TRỊ':>12}")
    print("-" * 38)
    try:
        for name in YAW_PARAMS:
            val = read_param(master, name)
            val_str = f"{val:12.4f}" if val is not None else "    (không có)"
            print(f"{name:<22} | {val_str}")
    finally:
        master.close()


if __name__ == "__main__":
    main()
