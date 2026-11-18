#!/usr/bin/env python3
import argparse
from pathlib import Path
import threading
import time
from typing import Any, Optional
import cv2

from camera import create_camera
from config_loader import PROJECT_ROOT, load_config, resolve_project_path
from detector import DetectedPerson, HumanDetector
from mavlink_client import MAVLinkClient, VALID_CONTROL_METHODS
from metrics import DebugMetrics
from tracker import TargetTracker
from visualizer import TelemetryStats, TrackingVisualizer
from web_server import TrackingWebServer

WINDOW_NAME = "Tricopter Vision Tracking"


class DroneTrackerApp:
    """
    Ứng dụng điều phối chính dùng chung cho cả:
    1. Thực tế Raspberry Pi 5 (Camera v3) + Pixhawk 6C (--profile raspberry_pi)
    2. Mô phỏng Gazebo + ArduPilot SITL (--profile simulation)
    3. Kiểm thử nhanh không phần cứng (--profile mock)
    """
    def __init__(self, config: dict[str, Any], debug: bool = False, debug_dir: Optional[str] = None):
        self.cfg = config
        frame_cfg = config["frame"]
        cam_cfg = config["camera"]
        det_cfg = config["detector"]
        trk_cfg = config["tracker"]
        mav_cfg = config["mavlink"]
        web_cfg = config["web"]
        loop_cfg = config["loop"]

        self.mode = str(config["mode"])
        self.frame_width = int(frame_cfg["width"])
        self.frame_height = int(frame_cfg["height"])
        self.headless = bool(loop_cfg["headless"])
        self.target_fps = float(loop_cfg["target_fps"])
        self.min_frame_interval = (1.0 / self.target_fps) if self.target_fps > 0 else 0.0

        self.debug = debug
        self.show_debug = debug
        self.metrics: Optional[DebugMetrics] = None
        self._frame_idx = 0
        self._last_debug_print = 0.0
        self._tracker_lock = threading.Lock()
        self._latest_detections: list[DetectedPerson] = []

        # 1. Kết nối MAVLink (Pixhawk 6C UART hoặc ArduPilot SITL UDP)
        self.mavlink = MAVLinkClient(
            conn_str=str(mav_cfg["conn_str"]),
            baud=int(mav_cfg["baud"]),
            heartbeat_timeout=float(mav_cfg["heartbeat_timeout"]),
            dry_run=bool(mav_cfg["dry_run"]),
            control_method=str(mav_cfg["control_method"]),
            max_yaw_rate=float(trk_cfg["max_yaw_rate"]),
            nogps_thrust=float(mav_cfg["nogps_thrust"]),
            rc_yaw_center=int(mav_cfg["rc_yaw_center"]),
            rc_yaw_range=int(mav_cfg["rc_yaw_range"]),
        )
        if debug:
            if debug_dir is None:
                debug_dir = str(PROJECT_ROOT / "logs" / time.strftime("%Y%m%d_%H%M%S"))
            self.metrics = DebugMetrics(
                debug_dir,
                deadband_px=float(trk_cfg["yaw_deadband_px"]),
                hfov_deg=float(frame_cfg["hfov"]),
                frame_width=self.frame_width,
            )
            self.mavlink.on_attitude = self.metrics.on_attitude
            self.mavlink.clock_sync = self.metrics.clock_sync
            print(f"[*] [DEBUG] Ghi metrics vào: {debug_dir}")
        self.mavlink.connect()

        # 2. Khởi tạo Detector (YOLOv8n NCNN + ByteTrack), TargetTracker & Visualizer
        model_path = resolve_project_path(det_cfg["model_dir"])
        self.detector = HumanDetector(
            model_path=str(model_path),
            conf_thresh=float(det_cfg["conf_thresh"]),
            nms_thresh=float(det_cfg["nms_thresh"]),
            imgsz=int(det_cfg["imgsz"]),
            num_threads=int(det_cfg["ncnn_threads"]),
        )
        self.tracker = TargetTracker(
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            yaw_kp=float(trk_cfg["yaw_kp"]),
            yaw_kd=float(trk_cfg["yaw_kd"]),
            yaw_deadband_px=float(trk_cfg["yaw_deadband_px"]),
            max_yaw_rate=float(trk_cfg["max_yaw_rate"]),
            dist_kp=float(trk_cfg["dist_kp"]),
            dist_deadband_ratio=float(trk_cfg["dist_deadband_ratio"]),
            min_vx=float(trk_cfg["min_vx"]),
            max_vx=float(trk_cfg["max_vx"]),
            max_vx_accel=float(trk_cfg["max_vx_accel"]),
            max_vx_decel=float(trk_cfg["max_vx_decel"]),
            height_ema_alpha=float(trk_cfg["height_ema_alpha"]),
            lost_timeout=float(trk_cfg["lost_timeout"]),
            auto_lock=bool(loop_cfg["auto_lock"]),
        )
        self.visualizer = TrackingVisualizer(
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            yaw_deadband_px=self.tracker.yaw_deadband_px,
            terminal_log_interval=float(loop_cfg["terminal_log_interval"]),
        )

        # 3. Khởi tạo Web Server (nếu bật) để giám sát & chọn mục tiêu từ trình duyệt
        self.web_server: Optional[TrackingWebServer] = None
        if web_cfg["enabled"]:
            self.web_server = TrackingWebServer(
                host=str(web_cfg["host"]),
                port=int(web_cfg["port"]),
                frame_width=self.frame_width,
                frame_height=self.frame_height,
                jpeg_quality=int(web_cfg["jpeg_quality"]),
                on_select_point=self._select_target_at,
                on_reset_target=self._reset_target,
            )
            self.web_server.start()

        # 4. Khởi tạo nguồn Camera (PiCameraV3, GazeboCamera, hoặc MockCamera)
        self.camera = create_camera(
            mode=self.mode,
            image_topic=str(cam_cfg.get("topic", "/camera/image_raw")),
            width=self.frame_width,
            height=self.frame_height,
            fps=int(cam_cfg["fps"]),
            buffer_count=int(cam_cfg.get("buffer_count", 2)),
            rotate_180=bool(cam_cfg.get("rotate_180", True)),
        )
        self.camera.start()

    def _select_target_at(self, x: int, y: int) -> Optional[int]:
        with self._tracker_lock:
            selected_id = self.tracker.select_by_point(x, y, self._latest_detections)
            ref_h = self.tracker.ref_height_px
        if selected_id is not None:
            print(
                f"[+] Đã chọn mục tiêu ID #{selected_id} tại ({x}, {y}) | "
                f"Chiều cao chuẩn: {ref_h:.0f}px"
            )
        return selected_id

    def _reset_target(self):
        with self._tracker_lock:
            self.tracker.reset()
        print("[*] Đã hủy khóa mục tiêu -> Hover.")

    def _on_mouse_click(self, event, x, y, flags, param):
        """Click chuột trái vào người trên cửa sổ OpenCV để chọn mục tiêu và bay bám theo."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self._select_target_at(x, y)

    def run(self):
        print(
            f"[*] Bắt đầu vòng lặp tracking (profile={self.cfg.get('profile')}, "
            f"mode={self.mode}, headless={self.headless}, "
            f"method={self.mavlink.control_method})."
        )
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

                # 1. Phát hiện và định danh người bằng YOLOv8n NCNN + ByteTrack
                infer_start = time.monotonic()
                detections = self.detector.detect_and_track(frame)
                infer_end = time.monotonic()
                infer_latency_ms = (infer_end - infer_start) * 1000.0

                # 2. Tính lệnh điều khiển (Yaw + Tiến/Lùi) cho mục tiêu đã chọn
                with self._tracker_lock:
                    self._latest_detections = detections
                    cmd, active_target = self.tracker.update(detections)
                    locked_id = self.tracker.locked_id
                    ref_h = self.tracker.ref_height_px
                    cur_h = self.tracker.current_height_px
                    occ_s = self.tracker.get_occlusion_elapsed()
                    lost_to = self.tracker.lost_timeout

                # 3. Gửi lệnh tới Pixhawk 6C / ArduPilot SITL
                t_cmd_sent = self.mavlink.send_control_command(cmd)

                # 4. Cập nhật thống kê & Metrics
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
                        detections, locked_id, error_x_px, cmd.yaw_rate, cmd.vx,
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
                    locked_id=locked_id,
                    num_targets=len(detections),
                    error_x_px=error_x_px,
                    ref_height_px=ref_h,
                    current_height_px=cur_h,
                    occlusion_elapsed_s=occ_s,
                    lost_timeout_s=lost_to,
                    yaw_rate_deg_s=cmd.yaw_rate,
                    vx_m_s=cmd.vx,
                    debug_lines=debug_lines,
                )

                self.visualizer.log_to_terminal(stats)

                need_web = self.web_server is not None and self.web_server.has_clients
                if not self.headless or need_web:
                    self.visualizer.draw_hud(frame, detections, stats)

                if need_web:
                    self.web_server.publish_frame(frame)

                if not self.headless:
                    cv2.imshow(WINDOW_NAME, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("d") and self.metrics is not None:
                        self.show_debug = not self.show_debug

                if self.min_frame_interval > 0.0:
                    elapsed = time.monotonic() - loop_start
                    if elapsed < self.min_frame_interval:
                        time.sleep(self.min_frame_interval - elapsed)

        except KeyboardInterrupt:
            print("\n[*] Dừng chương trình bởi người dùng.")
        finally:
            if self.web_server is not None:
                self.web_server.stop()
            self.camera.stop()
            self.mavlink.close()
            if self.metrics is not None:
                print(self.metrics.close())
            if not self.headless:
                cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hệ thống Tricopter Vision Tracking (Raspberry Pi 5 + Pixhawk 6C & Gazebo Sim)"
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Profile cấu hình trong config/: raspberry_pi (real), simulation (sim), mock",
    )
    parser.add_argument(
        "--mode",
        choices=["sim", "real", "mock"],
        default=None,
        help="Ghi đè chế độ chạy camera ('real', 'sim', 'mock')",
    )
    parser.add_argument("--mavlink", default=None, help="Ghi đè chuỗi kết nối MAVLink")
    parser.add_argument("--baud", type=int, default=None, help="Ghi đè baudrate UART")
    parser.add_argument(
        "--control-method",
        choices=list(VALID_CONTROL_METHODS),
        default=None,
        help="Ghi đè phương thức điều khiển MAVLink",
    )
    parser.add_argument("--topic", default=None, help="Ghi đè topic ảnh camera Gazebo")
    parser.add_argument("--imgsz", type=int, default=None, help="Ghi đè kích thước ảnh YOLO NCNN")
    parser.add_argument("--camera-fps", type=int, default=None, help="Ghi đè FPS camera")
    parser.add_argument("--target-fps", type=float, default=None, help="Ghi đè giới hạn FPS vòng lặp")
    parser.add_argument("--no-rotate", action="store_true", help="Không xoay ảnh 180 độ cho PiCameraV3")
    parser.add_argument("--auto-lock", action="store_true", help="Tự động khóa mục tiêu ID nhỏ nhất")
    parser.add_argument("--no-auto-lock", action="store_true", help="Tắt tự động khóa mục tiêu")
    parser.add_argument("--lost-timeout", type=float, default=None, help="Thời gian chờ khi mất dấu (giây)")
    parser.add_argument("--headless", action="store_true", help="Chạy không mở cửa sổ OpenCV")
    parser.add_argument("--gui", action="store_true", help="Ép mở cửa sổ OpenCV ngay cả trên profile headless")
    parser.add_argument("--web", action="store_true", help="Bật Web MJPEG Stream")
    parser.add_argument("--no-web", action="store_true", help="Tắt Web MJPEG Stream")
    parser.add_argument("--port", type=int, default=None, help="Cổng Web Server (mặc định: 5000)")
    parser.add_argument("--debug", action="store_true", help="Bật đo đạc & lưu CSV/báo cáo vào logs/")
    parser.add_argument("--debug-dir", default=None, help="Thư mục lưu CSV/báo cáo debug")
    parser.add_argument("--hfov", type=float, default=None, help="FOV ngang camera (độ)")
    parser.add_argument("--dry-run-mavlink", action="store_true", help="Giả lập kết nối MAVLink")
    return parser.parse_args()


def _build_overrides(args) -> dict[str, Any]:
    ov: dict[str, Any] = {}
    if args.mode is not None:
        ov["mode"] = args.mode
    if args.hfov is not None:
        ov.setdefault("frame", {})["hfov"] = args.hfov
    if args.topic is not None:
        ov.setdefault("camera", {})["topic"] = args.topic
    if args.camera_fps is not None:
        ov.setdefault("camera", {})["fps"] = args.camera_fps
    if args.no_rotate:
        ov.setdefault("camera", {})["rotate_180"] = False
    if args.imgsz is not None:
        ov.setdefault("detector", {})["imgsz"] = args.imgsz
    if args.lost_timeout is not None:
        ov.setdefault("tracker", {})["lost_timeout"] = args.lost_timeout
    if args.mavlink is not None:
        ov.setdefault("mavlink", {})["conn_str"] = args.mavlink
    if args.baud is not None:
        ov.setdefault("mavlink", {})["baud"] = args.baud
    if args.control_method is not None:
        ov.setdefault("mavlink", {})["control_method"] = args.control_method
    if args.dry_run_mavlink:
        ov.setdefault("mavlink", {})["dry_run"] = True
    if args.web:
        ov.setdefault("web", {})["enabled"] = True
    if args.no_web:
        ov.setdefault("web", {})["enabled"] = False
    if args.port is not None:
        ov.setdefault("web", {})["port"] = args.port
    if args.target_fps is not None:
        ov.setdefault("loop", {})["target_fps"] = args.target_fps
    if args.headless:
        ov.setdefault("loop", {})["headless"] = True
    if args.gui:
        ov.setdefault("loop", {})["headless"] = False
    if args.auto_lock:
        ov.setdefault("loop", {})["auto_lock"] = True
    if args.no_auto_lock:
        ov.setdefault("loop", {})["auto_lock"] = False
    return ov


def main():
    args = parse_args()
    profile = args.profile or args.mode or "raspberry_pi"
    cfg = load_config(profile=profile, overrides=_build_overrides(args))
    app = DroneTrackerApp(config=cfg, debug=args.debug, debug_dir=args.debug_dir)
    app.run()


if __name__ == "__main__":
    main()
