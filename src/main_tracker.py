#!/usr/bin/env python3
import argparse
from pathlib import Path
import time
import cv2

from camera import create_camera
from controller import ControlCommand
from detector import DetectedPerson, HumanDetector
from mavlink_client import MAVLinkClient
from tracker import CombinedTracker
from visualizer import TelemetryStats, TrackingVisualizer

WINDOW_NAME = "Drone Human Tracking (Sim & Real)"


class DroneTrackerApp:
    """
    Ứng dụng điều phối chính dùng chung cho cả:
    1. Mô phỏng Gazebo + ArduPilot SITL (--mode sim)
    2. Thực tế Raspberry Pi 5 (Camera v3) + Pixhawk 6C (--mode real)
    """
    def __init__(
        self,
        mode: str = "sim",
        mavlink_conn_str: str = "udpin:0.0.0.0:14550",
        baud: int = 921600,
        image_topic: str = "/camera/image_raw",
        imgsz: int = 640,
        enable_distance_tracking: bool = False,
        lost_timeout: float = 2.0,
        headless: bool = False,
        dry_run_mavlink: bool = False,
    ):
        self.mode = mode
        self.headless = headless

        # 1. Khởi tạo kết nối MAVLink (ArduPilot SITL hoặc Pixhawk 6C)
        self.mavlink = MAVLinkClient(
            conn_str=mavlink_conn_str,
            baud=baud,
            dry_run=dry_run_mavlink,
        )
        self.mavlink.connect()

        # 2. Khởi tạo Detector (YOLOv8 + ByteTrack), CombinedTracker (Selector + Yaw + Distance) & Visualizer
        model_path = Path(__file__).resolve().parent / "models" / "yolov8n.pt"
        self.detector = HumanDetector(model_path=str(model_path), imgsz=imgsz)
        self.tracker = CombinedTracker(
            frame_width=640,
            frame_height=480,
            enable_distance_tracking=enable_distance_tracking,
            lost_timeout=lost_timeout,
        )
        self.visualizer = TrackingVisualizer(
            frame_width=640,
            frame_height=480,
            yaw_deadband=self.tracker.yaw_deadband,
            dist_deadband_px=self.tracker.dist_deadband_px,
            terminal_log_interval=0.5,
        )
        self._latest_detections: list[DetectedPerson] = []

        # 3. Khởi tạo nguồn Camera (GazeboCamera cho 'sim', PiCameraV3 cho 'real')
        self.camera = create_camera(
            mode=mode,
            image_topic=image_topic,
            width=640,
            height=480,
        )
        self.camera.start()

    def _on_mouse_click(self, event, x, y, flags, param):
        """
        Xử lý sự kiện click chuột trên cửa sổ OpenCV:
        - Chuột trái: Chọn và khóa người tại vị trí click (đồng thời lấy cự ly hiện tại làm chuẩn).
        - Chuột phải: Hủy khóa, đưa drone về trạng thái Hover chờ chọn lại.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_id = self.tracker.select_by_point(x, y, self._latest_detections)
            if selected_id is not None:
                ref_h = self.tracker.ref_height_px
                ref_str = f"{ref_h:.0f}px" if ref_h is not None else "auto"
                print(f"[+] Đã khóa mục tiêu ID: #{selected_id} tại ({x}, {y}) | Chiều cao chuẩn: {ref_str}")
            else:
                print(f"[!] Không có mục tiêu nào tại vị trí click ({x}, {y})")
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.tracker.unlock_target()
            print("[*] Đã hủy khóa mục tiêu -> Drone Hover chờ người dùng chọn lại.")

    def send_control_command(self, cmd: ControlCommand):
        """Gửi lệnh điều khiển tổng hợp (vx, vy, vz, yaw_rate) qua MAVLinkClient."""
        self.mavlink.send_control_command(cmd)

    def send_yaw_rate(self, yaw_rate_deg_s: float):
        """Chỉ gửi tốc độ góc quay Yaw."""
        self.mavlink.send_yaw_rate(yaw_rate_deg_s)

    def run(self):
        print(f"[*] Bắt đầu vòng lặp tracking (mode={self.mode}, headless={self.headless}).")
        if not self.headless:
            print("    - Click chuột trái vào người trên màn hình để khóa mục tiêu.")
            print("    - Click chuột phải hoặc nhấn 'c' / Space để dừng bám (Hover chờ chọn lại).")
            print("    - Nhấn 'Tab' hoặc 'n' để chuyển đổi giữa các mục tiêu.")
            print("    - Nhấn 'a' để tự động chọn người gần tâm & gần drone nhất.")
            print("    - Nhấn 'd' để BẬT/TẮT nhanh chế độ tiến/lùi giữ khoảng cách (Vx).")
            print("    - Nhấn 'r' để chốt lại khoảng cách hiện tại làm mốc chuẩn mới.")
            print("    - Nhấn '+' ('=') hoặc '-' để giảm/tăng khoảng cách giữ mục tiêu.")
            print("    - Nhấn 'q' để thoát.")
            cv2.namedWindow(WINDOW_NAME)
            cv2.setMouseCallback(WINDOW_NAME, self._on_mouse_click)

        try:
            while True:
                self.mavlink.send_heartbeat_if_needed()

                frame = self.camera.read_frame()
                if frame is None:
                    if not self.headless:
                        if (cv2.waitKey(10) & 0xFF) == ord("q"):
                            break
                    else:
                        time.sleep(0.01)
                    continue

                loop_start = time.monotonic()
                fps = self.visualizer.update_fps(now=loop_start)

                # 1. Phát hiện và định danh đa mục tiêu bằng YOLOv8 + ByteTrack
                infer_start = time.monotonic()
                detections = self.detector.detect_and_track(frame)
                infer_latency_ms = (time.monotonic() - infer_start) * 1000.0
                self._latest_detections = detections

                # 2. Cập nhật bộ chọn mục tiêu & tính lệnh điều khiển (Yaw + Distance)
                cmd, active_target = self.tracker.update_targets(detections)

                # 3. Gửi setpoint tới ArduPilot / Pixhawk 6C
                self.mavlink.send_control_command(cmd)

                # 4. Tổng hợp thông số Telemetry & Hiển thị
                total_latency_ms = (time.monotonic() - loop_start) * 1000.0
                error_x_px = (
                    active_target.center[0] - self.tracker.center_x
                    if active_target is not None
                    else None
                )
                box_h_ratio = (
                    (active_target.bbox[3] - active_target.bbox[1]) / 480.0
                    if active_target is not None
                    else None
                )

                stats = TelemetryStats(
                    fps=fps,
                    infer_latency_ms=infer_latency_ms,
                    total_latency_ms=total_latency_ms,
                    state=self.tracker.state,
                    locked_id=self.tracker.locked_id,
                    num_targets=len(detections),
                    error_x_px=error_x_px,
                    ref_height_px=self.tracker.ref_height_px,
                    current_height_px=self.tracker.current_height_px,
                    error_height_px=self.tracker.error_height_px,
                    dist_deadband_px=self.tracker.dist_deadband_px,
                    box_height_ratio=box_h_ratio,
                    occlusion_elapsed_s=self.tracker.get_occlusion_elapsed(),
                    lost_timeout_s=self.tracker.lost_timeout,
                    yaw_rate_deg_s=cmd.yaw_rate,
                    vx_m_s=cmd.vx,
                    distance_tracking_enabled=self.tracker.enable_distance_tracking,
                )

                self.visualizer.log_to_terminal(stats)

                if not self.headless:
                    self.visualizer.draw_hud(frame, detections, stats)
                    cv2.imshow(WINDOW_NAME, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key in (9, ord("n")):  # Phím Tab hoặc 'n'
                        new_id = self.tracker.cycle_target(detections)
                        if new_id is not None:
                            print(f"[+] Đã chuyển sang mục tiêu ID: #{new_id}")
                    elif key == ord("a"):
                        self.tracker.trigger_auto_select()
                        print("[*] Đã bật chế độ tự động chọn mục tiêu tối ưu.")
                    elif key in (ord("c"), ord(" ")):
                        self.tracker.unlock_target()
                        print("[*] Đã hủy khóa -> Drone Hover chờ người dùng chọn mục tiêu.")
                    elif key == ord("d"):
                        enabled = self.tracker.toggle_distance_tracking()
                        if enabled and active_target is not None:
                            self.tracker.recalibrate_distance(active_target)
                        state_str = "BẬT (Yaw + Tiến/Lùi)" if enabled else "TẮT (Chỉ xoay Yaw)"
                        print(f"[*] Chế độ điều khiển khoảng cách (Vx): {state_str}")
                    elif key == ord("r"):
                        new_ref = self.tracker.recalibrate_distance(active_target)
                        if new_ref is not None:
                            print(f"[*] Đã chốt lại khoảng cách chuẩn mới: ref_height = {new_ref:.0f}px")
                    elif key in (ord("+"), ord("=")):
                        new_ref = self.tracker.adjust_reference_height(10.0)
                        print(f"[*] Đã chỉnh khoảng cách chuẩn GẦN HƠN: ref_height = {new_ref:.0f}px")
                    elif key in (ord("-"), ord("_")):
                        new_ref = self.tracker.adjust_reference_height(-10.0)
                        print(f"[*] Đã chỉnh khoảng cách chuẩn XA HƠN: ref_height = {new_ref:.0f}px")

        except KeyboardInterrupt:
            print("\n[*] Dừng chương trình bởi người dùng.")
        finally:
            self.camera.stop()
            self.mavlink.close()
            if not self.headless:
                cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hệ thống Drone Tracking Con Người (Gazebo Sim & Raspberry Pi 5 + Pixhawk 6C)"
    )
    parser.add_argument(
        "--mode",
        choices=["sim", "real", "mock"],
        default="sim",
        help="Chế độ chạy: 'sim' (Gazebo) hoặc 'real' (Raspberry Pi 5 Camera v3)",
    )
    parser.add_argument(
        "--mavlink",
        default=None,
        help="Chuỗi kết nối MAVLink (mặc định: 'udpin:0.0.0.0:14550' cho sim, '/dev/ttyAMA0' cho real)",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=921600,
        help="Tốc độ baudrate khi kết nối UART với Pixhawk 6C (mặc định: 921600)",
    )
    parser.add_argument(
        "--topic",
        default="/camera/image_raw",
        help="Topic ảnh camera trong Gazebo (chế độ sim)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Kích thước ảnh đầu vào YOLO (mặc định: 640 cho sim, 320 cho Raspberry Pi 5)",
    )
    parser.add_argument(
        "--distance",
        action="store_true",
        help="Bật sẵn chế độ điều khiển tiến/lùi giữ khoảng cách (Vx) cùng với góc Yaw",
    )
    parser.add_argument(
        "--lost-timeout",
        type=float,
        default=2.0,
        help="Thời gian (giây) chờ khi mục tiêu bị khuất trước khi chuyển sang Hover đợi lệnh (mặc định: 2.0s)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Chạy không mở cửa sổ OpenCV (dùng khi SSH vào Raspberry Pi 5)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Tự động chọn cấu hình mặc định tối ưu theo chế độ 'sim' hoặc 'real'
    mav_conn = args.mavlink
    if mav_conn is None:
        mav_conn = "udpin:0.0.0.0:14550" if args.mode in ("sim", "mock") else "/dev/ttyAMA0"

    yolo_imgsz = args.imgsz
    if yolo_imgsz is None:
        yolo_imgsz = 640 if args.mode == "sim" else 320

    app = DroneTrackerApp(
        mode=args.mode,
        mavlink_conn_str=mav_conn,
        baud=args.baud,
        image_topic=args.topic,
        imgsz=yolo_imgsz,
        enable_distance_tracking=args.distance,
        lost_timeout=args.lost_timeout,
        headless=args.headless,
    )
    app.run()
