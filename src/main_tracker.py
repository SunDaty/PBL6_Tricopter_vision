#!/usr/bin/env python3
import argparse
from pathlib import Path
import time
import cv2

from camera import create_camera
from detector import DetectedPerson, HumanDetector
from mavlink_client import MAVLinkClient
from metrics import DebugMetrics
from tracker import TargetTracker
from visualizer import TelemetryStats, TrackingVisualizer

WINDOW_NAME = "Tricopter Vision Tracking"


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
        lost_timeout: float = 2.0,
        headless: bool = False,
        dry_run_mavlink: bool = False,
        debug: bool = False,
        debug_dir: str = None,
        hfov: float = 66.0,
    ):
        self.mode = mode
        self.headless = headless
        self.debug = debug
        self.show_debug = debug
        self.metrics = None
        self._frame_idx = 0
        self._last_debug_print = 0.0

        # 1. Kết nối MAVLink (ArduPilot SITL hoặc Pixhawk 6C)
        self.mavlink = MAVLinkClient(
            conn_str=mavlink_conn_str,
            baud=baud,
            dry_run=dry_run_mavlink,
        )
        if debug:
            if debug_dir is None:
                debug_dir = str(
                    Path(__file__).resolve().parent / "debug_metrics" / time.strftime("%Y%m%d_%H%M%S")
                )
            self.metrics = DebugMetrics(debug_dir, deadband_px=15.0, hfov_deg=hfov, frame_width=640)
            self.mavlink.on_attitude = self.metrics.on_attitude
            self.mavlink.clock_sync = self.metrics.clock_sync
            print(f"[*] [DEBUG] Ghi metrics vào: {debug_dir}")
        self.mavlink.connect()

        # 2. Khởi tạo Detector (YOLOv8n + ByteTrack), TargetTracker & Visualizer
        model_path = Path(__file__).resolve().parent.parent / "models" / "yolov8n.pt"
        self.detector = HumanDetector(model_path=str(model_path), imgsz=imgsz)
        self.tracker = TargetTracker(
            frame_width=640,
            frame_height=480,
            lost_timeout=lost_timeout,
        )
        self.visualizer = TrackingVisualizer(
            frame_width=640,
            frame_height=480,
            yaw_deadband_px=self.tracker.yaw_deadband_px,
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
        """Click chuột trái vào người trên màn hình để chọn mục tiêu và bay bám theo."""
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_id = self.tracker.select_by_point(x, y, self._latest_detections)
            if selected_id is not None:
                print(
                    f"[+] Đã chọn mục tiêu ID #{selected_id} tại ({x}, {y}) | "
                    f"Chiều cao chuẩn: {self.tracker.ref_height_px:.0f}px"
                )

    def run(self):
        print(f"[*] Bắt đầu vòng lặp tracking (mode={self.mode}, headless={self.headless}).")
        if not self.headless:
            print("    - Click chuột trái vào người trên cửa sổ Camera để chọn mục tiêu và bay theo.")
            print("    - Nhấn 'q' để dừng drone (Hover) và thoát.")
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
                t_capture = self.camera.last_capture_time or loop_start
                fps = self.visualizer.update_fps(now=loop_start)

                # 1. Phát hiện và định danh người bằng YOLOv8n + ByteTrack
                infer_start = time.monotonic()
                detections = self.detector.detect_and_track(frame)
                infer_end = time.monotonic()
                infer_latency_ms = (infer_end - infer_start) * 1000.0
                self._latest_detections = detections

                # 2. Tính lệnh điều khiển (Yaw + Tiến/Lùi) cho mục tiêu đã chọn
                cmd, active_target = self.tracker.update(detections)

                # 3. Gửi lệnh vận tốc tới ArduPilot / Pixhawk 6C
                t_cmd_sent = self.mavlink.send_control_command(cmd)

                # 4. Hiển thị HUD & Telemetry
                total_latency_ms = (time.monotonic() - t_capture) * 1000.0
                error_x_px = (
                    active_target.center[0] - self.tracker.center_x
                    if active_target is not None
                    else None
                )

                debug_lines = None
                if self.metrics is not None:
                    self.metrics.on_frame(
                        self._frame_idx, t_capture, infer_start, infer_end, t_cmd_sent,
                        detections, self.tracker.locked_id, error_x_px, cmd.yaw_rate, cmd.vx,
                    )
                    self._frame_idx += 1
                    lines = self.metrics.hud_lines()
                    if self.show_debug:
                        debug_lines = lines
                    if loop_start - self._last_debug_print >= 1.0:
                        self._last_debug_print = loop_start
                        print("[DEBUG] " + " | ".join(lines))

                stats = TelemetryStats(
                    fps=fps,
                    infer_latency_ms=infer_latency_ms,
                    total_latency_ms=total_latency_ms,
                    locked_id=self.tracker.locked_id,
                    num_targets=len(detections),
                    error_x_px=error_x_px,
                    ref_height_px=self.tracker.ref_height_px,
                    current_height_px=self.tracker.current_height_px,
                    occlusion_elapsed_s=self.tracker.get_occlusion_elapsed(),
                    lost_timeout_s=self.tracker.lost_timeout,
                    yaw_rate_deg_s=cmd.yaw_rate,
                    vx_m_s=cmd.vx,
                    debug_lines=debug_lines,
                )

                self.visualizer.log_to_terminal(stats)

                if not self.headless:
                    self.visualizer.draw_hud(frame, detections, stats)
                    cv2.imshow(WINDOW_NAME, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("d") and self.metrics is not None:
                        self.show_debug = not self.show_debug

        except KeyboardInterrupt:
            print("\n[*] Dừng chương trình bởi người dùng.")
        finally:
            self.camera.stop()
            self.mavlink.close()
            if self.metrics is not None:
                print(self.metrics.close())
            if not self.headless:
                cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hệ thống Tricopter Vision Tracking (Gazebo Sim & Raspberry Pi 5 + Pixhawk 6C)"
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
        "--lost-timeout",
        type=float,
        default=2.0,
        help="Thời gian (giây) chờ khi mục tiêu bị khuất trước khi hủy khóa về Hover (mặc định: 2.0s)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Chạy không mở cửa sổ OpenCV",
    )
    parser.add_argument("--debug", action="store_true", help="Bật đo đạc: FPS, độ trễ, ID, overshoot yaw, E2E (HUD+CSV+báo cáo)")
    parser.add_argument("--debug-dir", default=None, help="Thư mục lưu CSV/báo cáo debug")
    parser.add_argument("--hfov", type=float, default=66.0, help="FOV ngang camera (độ) để quy đổi sai số pixel -> góc")
    parser.add_argument("--dry-run-mavlink", action="store_true", help="Không kết nối MAVLink thật (giả lập attitude khi --debug)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

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
        lost_timeout=args.lost_timeout,
        headless=args.headless,
        dry_run_mavlink=args.dry_run_mavlink,
        debug=args.debug,
        debug_dir=args.debug_dir,
        hfov=args.hfov,
    )
    app.run()
