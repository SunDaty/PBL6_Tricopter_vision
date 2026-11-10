# src/visualizer.py
from dataclasses import dataclass
import time
from typing import Optional
import cv2
import numpy as np

from detector import DetectedPerson
from selector import TrackingState


@dataclass
class TelemetryStats:
    """Tập hợp các thông số đo đạc thời gian thực để hiển thị lên HUD và Terminal."""
    fps: float = 0.0
    infer_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    state: TrackingState = TrackingState.AUTO_SEARCH
    locked_id: Optional[int] = None
    num_targets: int = 0
    error_x_px: Optional[float] = None
    ref_height_px: Optional[float] = None
    current_height_px: Optional[float] = None
    error_height_px: Optional[float] = None
    dist_deadband_px: float = 12.0
    box_height_ratio: Optional[float] = None
    occlusion_elapsed_s: float = 0.0
    lost_timeout_s: float = 2.0
    yaw_rate_deg_s: float = 0.0
    vx_m_s: float = 0.0
    distance_tracking_enabled: bool = False


class TrackingVisualizer:
    """
    Module chuyên trách:
    1. Tính toán FPS, độ trễ suy luận YOLO (Inference Latency) và độ trễ vòng lặp (Loop Latency).
    2. Vẽ HUD trực quan lên màn hình Camera:
       - Vùng Deadband góc Yaw (±15px) ở tâm khung hình.
       - Vùng Deadband Tiến/Lùi (±12px) và Thước đo khoảng cách (Distance Gauge Bar) cạnh mục tiêu.
       - ID từng người, vector sai số và bảng thông số Telemetry.
    3. Xuất log thông số quan trọng ra màn hình Terminal theo định kỳ.
    """
    def __init__(
        self,
        frame_width: int = 640,
        frame_height: int = 480,
        yaw_deadband: float = 15.0,
        dist_deadband_px: float = 12.0,
        terminal_log_interval: float = 0.5,
        fps_alpha: float = 0.15,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.center_x = int(frame_width / 2)
        self.center_y = int(frame_height / 2)
        self.yaw_deadband = int(yaw_deadband)
        self.dist_deadband_px = float(dist_deadband_px)
        self.terminal_log_interval = terminal_log_interval
        self.fps_alpha = fps_alpha

        self._prev_frame_time: Optional[float] = None
        self._smoothed_fps: float = 0.0
        self._last_terminal_log_time: float = 0.0

    def update_fps(self, now: Optional[float] = None) -> float:
        """Cập nhật và làm mượt chỉ số FPS bằng bộ lọc EMA."""
        current_time = now if now is not None else time.monotonic()
        if self._prev_frame_time is not None:
            dt = current_time - self._prev_frame_time
            if dt > 1e-6:
                instant_fps = 1.0 / dt
                if self._smoothed_fps == 0.0:
                    self._smoothed_fps = instant_fps
                else:
                    self._smoothed_fps = (
                        self.fps_alpha * instant_fps
                        + (1.0 - self.fps_alpha) * self._smoothed_fps
                    )
        self._prev_frame_time = current_time
        return self._smoothed_fps

    def _draw_yaw_deadband(self, frame: np.ndarray):
        """
        Vẽ đường tâm khung hình và vùng chết góc Yaw (Deadband Zone) [center_x - db, center_x + db].
        """
        left_db = max(0, self.center_x - self.yaw_deadband)
        right_db = min(self.frame_width - 1, self.center_x + self.yaw_deadband)

        # Tô phủ mờ vùng Deadband góc Yaw ở giữa màn hình
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (left_db, 28),
            (right_db, self.frame_height - 24),
            (255, 255, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.07, frame, 0.93, 0, frame)

        # Đường tâm chính giữa (nét liền màu Cyan)
        cv2.line(
            frame,
            (self.center_x, 28),
            (self.center_x, self.frame_height - 24),
            (255, 255, 0),
            1,
        )

        # Hai biên Deadband trái/phải (đường nét đứt màu Vàng/Cyan)
        step = 12
        for y in range(28, self.frame_height - 24, step * 2):
            y_end = min(y + step, self.frame_height - 24)
            cv2.line(frame, (left_db, y), (left_db, y_end), (0, 255, 255), 1)
            cv2.line(frame, (right_db, y), (right_db, y_end), (0, 255, 255), 1)

        # Dấu cộng tâm ngắm (Crosshair)
        cv2.line(
            frame,
            (self.center_x - 8, self.center_y),
            (self.center_x + 8, self.center_y),
            (255, 255, 0),
            1,
        )

        # Nhãn vùng Deadband góc Yaw
        db_label = f"YAW DEADBAND (+/-{self.yaw_deadband}px)"
        cv2.putText(
            frame,
            db_label,
            (self.center_x - 78, self.frame_height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 255, 255),
            1,
        )

    def _draw_distance_deadband_and_gauge(
        self,
        frame: np.ndarray,
        person: DetectedPerson,
        stats: TelemetryStats,
    ):
        """
        Vẽ trực quan vùng Deadband Tiến/Lùi (Distance Deadband) và Thước đo khoảng cách
        ngay bên cạnh mục tiêu đang khóa:
        1. Vạch giới hạn chiều cao chuẩn (ref_height_px ± dist_deadband_px) trên khung bao người.
        2. Thước đo dọc (Vertical Gauge Bar) bên phải khung bao hiển thị vùng:
           - TOO CLOSE (Lùi lại: Vx < 0)
           - DIST DEADBAND (Đứng yên: Vx = 0)
           - TOO FAR (Tiến tới: Vx > 0)
        """
        if stats.ref_height_px is None or stats.current_height_px is None:
            return

        x1, y1, x2, y2 = person.bbox
        _, target_y = map(int, person.center)
        ref_h = stats.ref_height_px
        cur_h = stats.current_height_px
        db_px = stats.dist_deadband_px

        # 1. Vẽ khung/vạch chiều cao chuẩn (Reference Height ± Deadband) đối xứng qua tâm Y của người
        half_ref = ref_h / 2.0
        half_min = max(10.0, (ref_h - db_px) / 2.0)
        half_max = (ref_h + db_px) / 2.0

        ref_top = int(max(30, target_y - half_ref))
        ref_bot = int(min(self.frame_height - 26, target_y + half_ref))
        max_top = int(max(30, target_y - half_max))
        max_bot = int(min(self.frame_height - 26, target_y + half_max))
        min_top = int(max(30, target_y - half_min))
        min_bot = int(min(self.frame_height - 26, target_y + half_min))

        # Vẽ 2 dải Deadband ngang ở đỉnh và đáy mục tiêu (màu Cyan)
        cv2.line(frame, (x1 - 6, ref_top), (x2 + 6, ref_top), (255, 255, 0), 1)
        cv2.line(frame, (x1 - 6, ref_bot), (x2 + 6, ref_bot), (255, 255, 0), 1)
        cv2.rectangle(frame, (x1 - 4, max_top), (x2 + 4, min_top), (255, 255, 0), 1)
        cv2.rectangle(frame, (x1 - 4, min_bot), (x2 + 4, max_bot), (255, 255, 0), 1)

        # 2. Vẽ Thước đo Tiến/Lùi (Vertical Distance Gauge) bên phải khung bao mục tiêu
        gauge_x = x2 + 12 if x2 + 65 < self.frame_width else max(8, x1 - 58)
        gauge_y1 = max(36, target_y - 60)
        gauge_y2 = min(self.frame_height - 32, target_y + 60)
        gauge_h = max(40, gauge_y2 - gauge_y1)
        gauge_mid = int((gauge_y1 + gauge_y2) / 2)
        gauge_w = 12

        # Nền thước đo
        cv2.rectangle(
            frame,
            (gauge_x, gauge_y1),
            (gauge_x + gauge_w, gauge_y2),
            (30, 30, 30),
            -1,
        )

        # Vùng Deadband ở giữa thước đo (tương ứng ±dist_deadband_px, hiển thị màu Cyan)
        scale_px_to_bar = (gauge_h / 2.0) / max(30.0, db_px * 3.5)
        db_bar_half = max(6, int(db_px * scale_px_to_bar))
        db_y1 = max(gauge_y1, gauge_mid - db_bar_half)
        db_y2 = min(gauge_y2, gauge_mid + db_bar_half)

        # Tô màu 3 vùng trên thước: Trên = Quá gần (Lùi), Giữa = Deadband (Đứng yên), Dưới = Xa (Tiến)
        cv2.rectangle(frame, (gauge_x, gauge_y1), (gauge_x + gauge_w, db_y1), (0, 90, 220), -1)
        cv2.rectangle(frame, (gauge_x, db_y1), (gauge_x + gauge_w, db_y2), (180, 180, 0), -1)
        cv2.rectangle(frame, (gauge_x, db_y2), (gauge_x + gauge_w, gauge_y2), (0, 150, 0), -1)
        cv2.rectangle(frame, (gauge_x, gauge_y1), (gauge_x + gauge_w, gauge_y2), (200, 200, 200), 1)

        # Vạch chỉ thị chiều cao hiện tại (cur_h so với ref_h):
        # cur_h > ref_h (lại gần) -> vạch đi lên vùng Đỏ/Cam (BACK)
        # cur_h < ref_h (đi xa) -> vạch đi xuống vùng Xanh lá (FWD)
        diff_h = cur_h - ref_h
        marker_offset = int(diff_h * scale_px_to_bar)
        marker_y = max(gauge_y1 + 2, min(gauge_y2 - 2, gauge_mid - marker_offset))

        cv2.line(
            frame,
            (gauge_x - 3, marker_y),
            (gauge_x + gauge_w + 3, marker_y),
            (255, 255, 255),
            3,
        )

        # Nhãn trạng thái Tiến / Lùi / Giữ cự ly cạnh thước đo
        if abs(diff_h) <= db_px:
            action_txt = "HOLD (DB)"
            action_col = (255, 255, 0)
        elif diff_h > db_px:
            action_txt = "BACKWARD"
            action_col = (0, 120, 255)
        else:
            action_txt = "FORWARD"
            action_col = (0, 255, 0)

        cv2.putText(
            frame,
            action_txt,
            (gauge_x - 4, max(35, gauge_y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            action_col,
            1,
        )
        cv2.putText(
            frame,
            f"H:{cur_h:.0f}/{ref_h:.0f}px",
            (gauge_x - 4, min(self.frame_height - 26, gauge_y2 + 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (220, 220, 220),
            1,
        )

    def _draw_targets(
        self,
        frame: np.ndarray,
        detections: list[DetectedPerson],
        stats: TelemetryStats,
    ):
        """Vẽ khung bao, nhãn ID nổi bật, Deadband Tiến/Lùi và đường sai số cho từng người."""
        for person in detections:
            x1, y1, x2, y2 = person.bbox
            target_x, target_y = map(int, person.center)
            is_locked = (
                stats.state == TrackingState.LOCKED
                and person.track_id == stats.locked_id
            )

            if is_locked:
                # Khung bao xanh lá đậm cho mục tiêu đang khóa
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Vẽ điểm tâm mục tiêu và đoạn thẳng nối tới tâm màn hình (hiển thị sai số e_x)
                in_deadband = abs(target_x - self.center_x) < self.yaw_deadband
                err_color = (255, 255, 0) if in_deadband else (0, 140, 255)
                cv2.circle(frame, (target_x, target_y), 5, (0, 0, 255), -1)
                cv2.line(
                    frame,
                    (self.center_x, target_y),
                    (target_x, target_y),
                    err_color,
                    2,
                )

                # Vẽ vùng Deadband Tiến/Lùi và Thước đo khoảng cách
                self._draw_distance_deadband_and_gauge(frame, person, stats)

                # Nhãn ID có nền đặc nổi bật phía trên đầu mục tiêu
                id_badge = f"[LOCKED] ID #{person.track_id} ({person.conf:.2f})"
                (tw, th), _ = cv2.getTextSize(id_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
                badge_y1 = max(30, y1 - th - 10)
                cv2.rectangle(
                    frame,
                    (x1, badge_y1),
                    (x1 + tw + 8, badge_y1 + th + 8),
                    (0, 140, 0),
                    -1,
                )
                cv2.putText(
                    frame,
                    id_badge,
                    (x1 + 4, badge_y1 + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 255),
                    2,
                )

                # Hiển thị sai số pixel ngay cạnh đường nối tâm
                err_px = target_x - self.center_x
                cv2.putText(
                    frame,
                    f"ex: {err_px:+.0f}px",
                    (min(self.center_x, target_x) + 5, max(45, target_y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    err_color,
                    1,
                )
            else:
                # Khung bao màu vàng cho các người khác kèm ID rõ ràng
                color = (0, 215, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                cv2.circle(frame, (target_x, target_y), 3, color, -1)

                id_badge = f"ID #{person.track_id} [Click]"
                (tw, th), _ = cv2.getTextSize(id_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                badge_y1 = max(30, y1 - th - 8)
                cv2.rectangle(
                    frame,
                    (x1, badge_y1),
                    (x1 + tw + 6, badge_y1 + th + 6),
                    (30, 30, 30),
                    -1,
                )
                cv2.putText(
                    frame,
                    id_badge,
                    (x1 + 3, badge_y1 + th + 1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    color,
                    1,
                )

    def _draw_telemetry_overlay(self, frame: np.ndarray, stats: TelemetryStats):
        """Vẽ bảng thông số kỹ thuật (FPS, Latency, Sai số Yaw/Dist, Vận tốc) ở góc trái trên."""
        # 1. Thanh trạng thái (Top Banner)
        if stats.state == TrackingState.WAITING_USER_SELECT:
            status_text = (
                "STATE: WAITING USER SELECT (HOVER) - Click person or press [TAB] / [A]"
            )
            status_color = (0, 0, 255)
        elif stats.state == TrackingState.LOCKED and stats.error_x_px is None:
            status_text = (
                f"STATE: ID #{stats.locked_id} OCCLUDED "
                f"({stats.occlusion_elapsed_s:.1f}s / {stats.lost_timeout_s:.1f}s) - HOVERING"
            )
            status_color = (0, 165, 255)
        elif stats.state == TrackingState.LOCKED:
            dist_state = "YAW + DIST (Vx)" if stats.distance_tracking_enabled else "YAW ONLY"
            status_text = f"STATE: LOCKED ID #{stats.locked_id} [{dist_state}]"
            status_color = (0, 255, 0)
        else:
            status_text = "STATE: AUTO SEARCHING TARGET..."
            status_color = (255, 255, 0)

        cv2.rectangle(frame, (0, 0), (self.frame_width, 28), (20, 20, 20), -1)
        cv2.putText(
            frame,
            status_text,
            (10, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            status_color,
            2,
        )

        # 2. Hộp Telemetry góc trên bên trái
        panel_x, panel_y = 8, 34
        panel_w, panel_h = 268, 136
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (15, 15, 15),
            -1,
        )
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (100, 100, 100),
            1,
        )

        locked_str = f"#{stats.locked_id}" if stats.locked_id is not None else "None"
        err_x_str = f"{stats.error_x_px:+.0f}px" if stats.error_x_px is not None else "N/A"
        if stats.current_height_px is not None and stats.ref_height_px is not None:
            dist_str = (
                f"H:{stats.current_height_px:.0f}/{stats.ref_height_px:.0f}px "
                f"(DB:+/-{stats.dist_deadband_px:.0f})"
            )
        else:
            dist_str = f"N/A (DB:+/-{stats.dist_deadband_px:.0f}px)"

        dist_mode_str = "ON" if stats.distance_tracking_enabled else "OFF"

        lines = [
            f"FPS: {stats.fps:.1f} | Latency: {stats.total_latency_ms:.1f} ms",
            f"YOLO+ByteTrack: {stats.infer_latency_ms:.1f} ms",
            f"Persons: {stats.num_targets} | Locked ID: {locked_str}",
            f"Yaw Err : {err_x_str} (DB:+/-{self.yaw_deadband}px)",
            f"Dist Box: {dist_str}",
            f"YawRate : {stats.yaw_rate_deg_s:+.2f} deg/s",
            f"Vel X   : {stats.vx_m_s:+.2f} m/s [Dist: {dist_mode_str}]",
        ]

        for i, line in enumerate(lines):
            y_pos = panel_y + 17 + i * 18
            color = (0, 255, 0) if i in (0, 5, 6) else (230, 230, 230)
            cv2.putText(
                frame,
                line,
                (panel_x + 8, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.41,
                color,
                1,
            )

        # 3. Thanh hướng dẫn phím tắt ở đáy màn hình
        help_text = "Click:Lock | C:Hover | TAB:Next | D:Toggle Vx | R:SetDist | +/-:Dist | Q:Quit"
        cv2.rectangle(
            frame,
            (0, self.frame_height - 24),
            (self.frame_width, self.frame_height),
            (20, 20, 20),
            -1,
        )
        cv2.putText(
            frame,
            help_text,
            (8, self.frame_height - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (215, 215, 215),
            1,
        )

    def draw_hud(
        self,
        frame: np.ndarray,
        detections: list[DetectedPerson],
        stats: TelemetryStats,
    ) -> np.ndarray:
        """Vẽ toàn bộ giao diện HUD (Deadband Yaw & Tiến/Lùi, ID người, bảng Telemetry) lên frame."""
        self._draw_yaw_deadband(frame)
        self._draw_targets(frame, detections, stats)
        self._draw_telemetry_overlay(frame, stats)
        return frame

    def log_to_terminal(self, stats: TelemetryStats, now: Optional[float] = None):
        """In thông số quan trọng ra màn hình Terminal theo chu kỳ (mặc định 0.5s/lần)."""
        current_time = now if now is not None else time.monotonic()
        if current_time - self._last_terminal_log_time < self.terminal_log_interval:
            return
        self._last_terminal_log_time = current_time

        locked_str = f"#{stats.locked_id}" if stats.locked_id is not None else "None"
        err_str = f"{stats.error_x_px:+6.1f}px" if stats.error_x_px is not None else "   N/A "
        if stats.current_height_px is not None and stats.ref_height_px is not None:
            dist_h_str = f"{stats.current_height_px:3.0f}/{stats.ref_height_px:3.0f}px"
        else:
            dist_h_str = "   N/A   "

        if stats.state == TrackingState.LOCKED and stats.error_x_px is None:
            state_str = f"OCCLUDED({stats.occlusion_elapsed_s:.1f}s)"
        else:
            state_str = stats.state.value

        print(
            f"[TELEMETRY] FPS: {stats.fps:5.1f} | "
            f"Latency: {stats.total_latency_ms:5.1f}ms (AI: {stats.infer_latency_ms:5.1f}ms) | "
            f"State: {state_str:<19} | "
            f"Locked: {locked_str:<4} | "
            f"ErrX: {err_str} | "
            f"BoxH: {dist_h_str} | "
            f"YawRate: {stats.yaw_rate_deg_s:+6.2f} deg/s | "
            f"Vx: {stats.vx_m_s:+5.2f} m/s"
        )


if __name__ == "__main__":
    print("=== Kiểm thử độc lập Module 9: visualizer.py (TrackingVisualizer) ===")
    vis = TrackingVisualizer(yaw_deadband=15.0, dist_deadband_px=12.0)
    vis.update_fps(now=0.0)
    fps = vis.update_fps(now=0.033)
    p1 = DetectedPerson(track_id=1, bbox=(360, 120, 440, 300), conf=0.95)
    stats = TelemetryStats(
        fps=fps,
        infer_latency_ms=17.5,
        total_latency_ms=20.0,
        state=TrackingState.LOCKED,
        locked_id=1,
        num_targets=1,
        error_x_px=80.0,
        ref_height_px=140.0,
        current_height_px=180.0,
        error_height_px=-40.0,
        dist_deadband_px=12.0,
        yaw_rate_deg_s=6.4,
        vx_m_s=-0.60,
        distance_tracking_enabled=True,
    )
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    out = vis.draw_hud(canvas, [p1], stats)
    assert out.shape == (480, 640, 3) and out.sum() > 0
    vis.log_to_terminal(stats, now=1.0)
    print("[PASS] Vẽ Deadband Yaw, Thước đo Deadband Tiến/Lùi, ID người và xuất log Terminal thành công!")
    print("=== Module visualizer.py hoạt động tốt! ===")
