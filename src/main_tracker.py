#!/usr/bin/env python3
import argparse
import math
import threading
import time
from typing import Any, Optional
import cv2
import numpy as np

from camera import create_camera
from config_loader import PROJECT_ROOT, load_config, resolve_project_path
from detector import DetectedPerson, HumanDetector
from mavlink_client import MAVLinkClient, VALID_CONTROL_METHODS
from metrics import DebugMetrics
from tracker import TargetTracker
from visualizer import TelemetryStats, TrackingVisualizer

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
        telem_cfg = config.get("telemetry", {})
        web_cfg = config["web"]
        loop_cfg = config["loop"]

        self.mode = str(config["mode"])
        self.frame_width = int(frame_cfg["width"])
        self.frame_height = int(frame_cfg["height"])
        self.hfov_deg = float(frame_cfg["hfov"])
        self.headless = bool(loop_cfg["headless"])
        self.target_fps = float(loop_cfg["target_fps"])
        self.min_frame_interval = (1.0 / self.target_fps) if self.target_fps > 0 else 0.0
        self.ego_motion_comp = bool(det_cfg.get("ego_motion_comp", True))

        self.debug = debug
        self.show_debug = debug
        self.metrics: Optional[DebugMetrics] = None
        self._frame_idx = 0
        self._run_start_time: Optional[float] = None
        self._tracker_lock = threading.Lock()
        self._latest_detections: list[DetectedPerson] = []
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_psi_cap: Optional[float] = None

        # 1. Kết nối MAVLink (Pixhawk 6C UART hoặc ArduPilot SITL UDP) + Telemetry ATTITUDE
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
            telemetry_enabled=bool(telem_cfg.get("enabled", True)),
            attitude_hz=int(telem_cfg.get("attitude_hz", 50)),
        )
        if debug:
            if debug_dir is None:
                debug_dir = str(PROJECT_ROOT / "logs" / time.strftime("%Y%m%d_%H%M%S"))
            self.metrics = DebugMetrics(
                debug_dir,
                deadband_px=float(trk_cfg["yaw_deadband_px"]),
                hfov_deg=self.hfov_deg,
                frame_width=self.frame_width,
            )
            self.mavlink.on_attitude = self.metrics.on_attitude
            self.mavlink.clock_sync = self.metrics.clock_sync
            print(f"[*] [DEBUG] Ghi metrics vào: {debug_dir}")
        self.mavlink.connect()

        # 2. Khởi tạo Detector (YOLOv8n NCNN + ByteTrack bù ego-motion), TargetTracker & Visualizer
        model_path = resolve_project_path(det_cfg["model_dir"])
        self.detector = HumanDetector(
            model_path=str(model_path),
            conf_thresh=float(det_cfg["conf_thresh"]),
            nms_thresh=float(det_cfg["nms_thresh"]),
            imgsz=int(det_cfg["imgsz"]),
            num_threads=int(det_cfg["ncnn_threads"]),
            track_activation_threshold=float(det_cfg.get("track_activation_threshold", 0.45)),
            lost_track_buffer=int(det_cfg.get("lost_track_buffer", 30)),
            minimum_matching_threshold=float(det_cfg.get("minimum_matching_threshold", 0.85)),
            frame_width=self.frame_width,
            hfov_deg=self.hfov_deg,
        )

        reacq_cfg = trk_cfg.get("reacquire", {})
        h_ratio = reacq_cfg.get("height_ratio", [0.65, 1.5])
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
            hfov_deg=self.hfov_deg,
            yaw_deadband_exit_px=(
                float(trk_cfg["yaw_deadband_exit_px"])
                if trk_cfg.get("yaw_deadband_exit_px") is not None
                else None
            ),
            max_yaw_accel=float(trk_cfg.get("max_yaw_accel", 60.0)),
            latency_comp=bool(trk_cfg.get("latency_comp", True)),
            coast_timeout=float(trk_cfg.get("coast_timeout", 0.6)),
            reacquire_enabled=bool(reacq_cfg.get("enabled", True)),
            reacquire_max_bearing_deg=float(reacq_cfg.get("max_bearing_deg", 12.0)),
            reacquire_height_ratio=(float(h_ratio[0]), float(h_ratio[1])),
            reacquire_use_color_hist=bool(reacq_cfg.get("use_color_hist", True)),
            reacquire_hist_min_corr=float(reacq_cfg.get("hist_min_corr", 0.45)),
            single_person_fallback=bool(reacq_cfg.get("single_person_fallback", True)),
        )
        self.visualizer = TrackingVisualizer(
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            yaw_deadband_px=self.tracker.yaw_deadband_px,
            terminal_log_interval=float(loop_cfg.get("terminal_log_interval", 0.0)),
            hud_mode=str(loop_cfg.get("hud_mode", "minimal")),
        )

        # 3. Khởi tạo Web Server (nếu bật) để giám sát & chọn mục tiêu từ trình duyệt
        self.web_server = None
        if web_cfg["enabled"]:
            from web_server import TrackingWebServer

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
            ae_exposure_mode=str(cam_cfg.get("ae_exposure_mode", "short")),
            af_mode=str(cam_cfg.get("af_mode", "manual")),
            lens_position=float(cam_cfg.get("lens_position", 0.3)),
        )
        self.camera.start()

    def _select_target_at(self, x: int, y: int) -> Optional[int]:
        with self._tracker_lock:
            selected_id = self.tracker.select_by_point(
                x,
                y,
                self._latest_detections,
                frame=self._latest_frame,
                psi_capture=self._latest_psi_cap,
            )
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

    def _print_startup_summary(self):
        print(
            "=== KHỞI ĐỘNG HỆ THỐNG BÁM MỤC TIÊU ===\n"
            f"  Profile: {self.cfg.get('profile')} | Mode: {self.mode} | "
            f"MAVLink: {self.mavlink.control_method} (telem={self.mavlink.telemetry_enabled}@{self.mavlink.attitude_hz}Hz)\n"
            f"  Yaw: kp={self.tracker.yaw_kp}, kd={self.tracker.yaw_kd}, "
            f"db={self.tracker.yaw_deadband_px:.0f}/{self.tracker.yaw_deadband_exit_px:.0f}px, "
            f"max={self.tracker.max_yaw_rate:.0f}deg/s, accel={self.tracker.yaw_slew.max_rate:.0f}deg/s²\n"
            f"  Bù trễ: {self.tracker.latency_comp} | Bù ego-motion: {self.ego_motion_comp} | "
            f"COAST: {self.tracker.coast_timeout:.1f}s | Lost: {self.tracker.lost_timeout:.1f}s | "
            f"RE-ACQ: {self.tracker.reacquire_enabled}\n"
            f"  HUD: {self.visualizer.hud_mode} | Log định kỳ: "
            f"{'TẮT' if self.visualizer.terminal_log_interval <= 0 else f'{self.visualizer.terminal_log_interval}s'}"
        )

    def run(self):
        self._print_startup_summary()
        if not self.headless:
            print("    - Click chuột trái vào người trên cửa sổ Camera để chọn mục tiêu và bay theo.")
            print("    - Nhấn 'q' để dừng drone (Hover) và thoát.")
            cv2.namedWindow(WINDOW_NAME)
            cv2.setMouseCallback(WINDOW_NAME, self._on_mouse_click)

        self._run_start_time = time.monotonic()

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

                # 1. Lấy góc yaw tại thời điểm chụp ảnh để bù ego-motion cho ByteTrack
                psi_cap = self.mavlink.yaw_history.yaw_at(t_capture)

                infer_start = time.monotonic()
                detections = self.detector.detect_and_track(
                    frame,
                    psi=psi_cap if self.ego_motion_comp else None,
                )
                infer_end = time.monotonic()
                infer_latency_ms = (infer_end - infer_start) * 1000.0

                # 2. Lấy góc yaw hiện tại (bù độ trễ ảnh->lệnh) và tính lệnh điều khiển
                psi_now = self.mavlink.yaw_history.latest(infer_end)
                with self._tracker_lock:
                    self._latest_detections = detections
                    self._latest_frame = frame
                    self._latest_psi_cap = psi_cap
                    cmd, active_target = self.tracker.update(
                        detections,
                        now=infer_end,
                        frame=frame,
                        psi_capture=psi_cap,
                        psi_now=psi_now,
                    )
                    locked_id = self.tracker.locked_id
                    ref_h = self.tracker.ref_height_px
                    cur_h = self.tracker.current_height_px
                    occ_s = self.tracker.get_occlusion_elapsed(now=infer_end)
                    lost_to = self.tracker.lost_timeout
                    trk_state = self.tracker.state
                    err_deg_now = self.tracker.last_error_deg_now
                    reacq_count = self.tracker.reacquire_count
                    reacquired = self.tracker.last_reacquired
                    lost_id = self.tracker.last_lost_id
                    pred_x_px = self.tracker.predicted_x_px

                # 3. Gửi lệnh tới Pixhawk 6C / ArduPilot SITL ngay lập tức
                t_cmd_sent = self.mavlink.send_control_command(cmd)

                # Log sự kiện rời rạc (không in lặp mỗi frame khi đang bay)
                if reacquired is not None:
                    print(f"[RE-ACQ] Tự bắt lại mục tiêu: #{reacquired[0]} -> #{reacquired[1]} (tổng: {reacq_count})")
                elif lost_id is not None:
                    print(f"[*] Mất dấu mục tiêu #{lost_id} quá {lost_to:.1f}s -> Hủy khóa & Hover.")

                # 4. Cập nhật Metrics (chỉ ghi CSV/thống kê, KHÔNG tính hud_lines định kỳ)
                total_latency_ms = (t_cmd_sent - t_capture) * 1000.0
                error_x_px = (
                    active_target.center[0] - self.tracker.center_x
                    if active_target is not None
                    else None
                )

                need_web = self.web_server is not None and self.web_server.has_clients
                need_draw = (not self.headless or need_web) and (self.visualizer.hud_mode != "off")

                debug_lines = None
                if self.metrics is not None:
                    self.metrics.on_frame(
                        self._frame_idx, t_capture, infer_start, infer_end, t_cmd_sent,
                        detections, locked_id, error_x_px, cmd.yaw_rate, cmd.vx,
                        state=trk_state,
                        err_deg_now=err_deg_now,
                        psi_cap_deg=math.degrees(psi_cap) if psi_cap is not None else None,
                        reacq_flag=(reacquired is not None),
                    )
                    if need_draw and self.visualizer.hud_mode == "full" and self.show_debug:
                        debug_lines = self.metrics.hud_lines()

                self._frame_idx += 1

                if need_draw or self.visualizer.terminal_log_interval > 0.0:
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
                        state=trk_state,
                        error_deg_now=err_deg_now,
                        reacquire_count=reacq_count,
                        predicted_x_px=pred_x_px,
                    )
                    self.visualizer.log_to_terminal(stats)
                    if need_draw:
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
            elif self._run_start_time is not None:
                dur = max(1e-6, time.monotonic() - self._run_start_time)
                print(
                    f"=== KẾT THÚC PHIÊN BAY ===\n"
                    f"  Thời lượng: {dur:.1f}s | Số khung: {self._frame_idx} | "
                    f"FPS TB: {self._frame_idx / dur:.1f} | Tự bắt lại (RE-ACQ): {self.tracker.reacquire_count}"
                )
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
    parser.add_argument(
        "--hud-mode",
        choices=["minimal", "full", "off"],
        default=None,
        help="Chế độ vẽ HUD trên video ('minimal', 'full', 'off')",
    )
    parser.add_argument("--no-ego-comp", action="store_true", help="Tắt bù chuyển động camera cho ByteTrack")
    parser.add_argument("--no-latency-comp", action="store_true", help="Tắt bù độ trễ yaw khi tính lệnh")
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
    if args.no_ego_comp:
        ov.setdefault("detector", {})["ego_motion_comp"] = False
    if args.lost_timeout is not None:
        ov.setdefault("tracker", {})["lost_timeout"] = args.lost_timeout
    if args.no_latency_comp:
        ov.setdefault("tracker", {})["latency_comp"] = False
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
    if args.hud_mode is not None:
        ov.setdefault("loop", {})["hud_mode"] = args.hud_mode
    return ov


def main():
    args = parse_args()
    profile = args.profile or args.mode or "raspberry_pi"
    cfg = load_config(profile=profile, overrides=_build_overrides(args))
    app = DroneTrackerApp(config=cfg, debug=args.debug, debug_dir=args.debug_dir)
    app.run()


if __name__ == "__main__":
    main()
