# src/visualizer.py
from dataclasses import dataclass
import time
from typing import Optional
import cv2
import numpy as np

from detector import DetectedPerson


@dataclass
class TelemetryStats:
    """Thông số đo đạc thời gian thực để hiển thị lên HUD và Terminal."""
    fps: float = 0.0
    infer_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    locked_id: Optional[int] = None
    num_targets: int = 0
    error_x_px: Optional[float] = None
    ref_height_px: Optional[float] = None
    current_height_px: Optional[float] = None
    occlusion_elapsed_s: float = 0.0
    lost_timeout_s: float = 2.0
    yaw_rate_deg_s: float = 0.0
    vx_m_s: float = 0.0


class TrackingVisualizer:
    """
    Module hiển thị giao diện HUD trực quan và gọn nhẹ lên cửa sổ Camera:
    1. Đường tâm ngắm và vùng chết góc Yaw (Deadband Zone).
    2. Khung bao từng người: Vàng (chưa chọn), Xanh lá (đang khóa bám theo).
    3. Bảng thông số Telemetry góc trên bên trái & thanh hướng dẫn tối giản ở đáy màn hình.
    """
    def __init__(
        self,
        frame_width: int = 640,
        frame_height: int = 480,
        yaw_deadband_px: float = 15.0,
        terminal_log_interval: float = 0.5,
        fps_alpha: float = 0.15,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.center_x = int(frame_width / 2)
        self.center_y = int(frame_height / 2)
        self.yaw_deadband = int(yaw_deadband_px)
        self.terminal_log_interval = terminal_log_interval
        self.fps_alpha = fps_alpha

        self._prev_frame_time: Optional[float] = None
        self._smoothed_fps: float = 0.0
        self._last_terminal_log_time: float = 0.0

    def update_fps(self, now: Optional[float] = None) -> float:
        """Cập nhật chỉ số FPS mượt bằng EMA."""
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

    def _draw_center_guide(self, frame: np.ndarray):
        """Vẽ đường tâm dọc và hai biên vùng chết góc Yaw."""
        left_db = max(0, self.center_x - self.yaw_deadband)
        right_db = min(self.frame_width - 1, self.center_x + self.yaw_deadband)

        cv2.line(frame, (self.center_x, 28), (self.center_x, self.frame_height - 24), (255, 255, 0), 1)
        cv2.line(frame, (left_db, 28), (left_db, self.frame_height - 24), (0, 200, 200), 1)
        cv2.line(frame, (right_db, 28), (right_db, self.frame_height - 24), (0, 200, 200), 1)
        cv2.line(frame, (self.center_x - 8, self.center_y), (self.center_x + 8, self.center_y), (255, 255, 0), 1)

    def _draw_targets(
        self,
        frame: np.ndarray,
        detections: list[DetectedPerson],
        stats: TelemetryStats,
    ):
        """Vẽ khung bao và nhãn ID cho từng người trong khung hình."""
        for person in detections:
            x1, y1, x2, y2 = person.bbox
            target_x, target_y = map(int, person.center)
            is_locked = stats.locked_id is not None and person.track_id == stats.locked_id

            if is_locked:
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (target_x, target_y), 5, (0, 0, 255), -1)
                cv2.line(frame, (self.center_x, target_y), (target_x, target_y), (0, 200, 255), 2)

                label = f"[LOCKED] ID #{person.track_id}"
                if stats.current_height_px is not None and stats.ref_height_px is not None:
                    label += f" | H:{stats.current_height_px:.0f}/{stats.ref_height_px:.0f}px"

                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                by1 = max(28, y1 - th - 8)
                cv2.rectangle(frame, (x1, by1), (x1 + tw + 8, by1 + th + 6), (0, 140, 0), -1)
                cv2.putText(
                    frame,
                    label,
                    (x1 + 4, by1 + th + 1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )
            else:
                color = (0, 215, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                label = f"ID #{person.track_id} [Click]"
                cv2.putText(
                    frame,
                    label,
                    (x1, max(40, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    color,
                    1,
                )

    def _draw_overlay(self, frame: np.ndarray, stats: TelemetryStats):
        """Vẽ thanh trạng thái trên cùng, hộp Telemetry góc trái và hướng dẫn ở đáy ảnh."""
        # 1. Thanh trạng thái trên cùng
        if stats.locked_id is None:
            status_text = "STATUS: HOVERING - Left-click a person to follow"
            status_color = (0, 215, 255)
        elif stats.error_x_px is None:
            status_text = (
                f"STATUS: ID #{stats.locked_id} OCCLUDED "
                f"({stats.occlusion_elapsed_s:.1f}s / {stats.lost_timeout_s:.1f}s) - HOVERING"
            )
            status_color = (0, 165, 255)
        else:
            status_text = f"STATUS: FOLLOWING ID #{stats.locked_id}"
            status_color = (0, 255, 0)

        cv2.rectangle(frame, (0, 0), (self.frame_width, 26), (20, 20, 20), -1)
        cv2.putText(frame, status_text, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, status_color, 2)

        # 2. Hộp Telemetry góc trên trái
        panel_x, panel_y, panel_w, panel_h = 8, 32, 245, 104
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        locked_str = f"#{stats.locked_id}" if stats.locked_id is not None else "None"
        err_x_str = f"{stats.error_x_px:+.0f}px" if stats.error_x_px is not None else "N/A"
        if stats.current_height_px is not None and stats.ref_height_px is not None:
            dist_str = f"{stats.current_height_px:.0f} / {stats.ref_height_px:.0f} px"
        else:
            dist_str = "N/A"

        lines = [
            f"FPS: {stats.fps:.1f} | AI: {stats.infer_latency_ms:.1f} ms",
            f"Persons: {stats.num_targets} | Locked: {locked_str}",
            f"Yaw Err: {err_x_str} | BoxH: {dist_str}",
            f"YawRate: {stats.yaw_rate_deg_s:+.2f} deg/s",
            f"Vel Vx : {stats.vx_m_s:+.2f} m/s",
        ]
        for i, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (panel_x + 8, panel_y + 18 + i * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 255, 0) if i in (0, 3, 4) else (230, 230, 230),
                1,
            )

        # 3. Thanh hướng dẫn tối giản dưới đáy
        cv2.rectangle(frame, (0, self.frame_height - 22), (self.frame_width, self.frame_height), (20, 20, 20), -1)
        cv2.putText(
            frame,
            "Left-Click: Select Target & Follow | Q: Quit",
            (10, self.frame_height - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (215, 215, 215),
            1,
        )

    def draw_hud(
        self,
        frame: np.ndarray,
        detections: list[DetectedPerson],
        stats: TelemetryStats,
    ) -> np.ndarray:
        """Vẽ toàn bộ HUD lên frame."""
        self._draw_center_guide(frame)
        self._draw_targets(frame, detections, stats)
        self._draw_overlay(frame, stats)
        return frame

    def log_to_terminal(self, stats: TelemetryStats, now: Optional[float] = None):
        """In thông số ra Terminal theo chu kỳ."""
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

        print(
            f"[TELEMETRY] FPS: {stats.fps:5.1f} | "
            f"AI: {stats.infer_latency_ms:5.1f}ms | "
            f"Locked: {locked_str:<4} | "
            f"ErrX: {err_str} | "
            f"BoxH: {dist_h_str} | "
            f"YawRate: {stats.yaw_rate_deg_s:+6.2f} deg/s | "
            f"Vx: {stats.vx_m_s:+5.2f} m/s"
        )


if __name__ == "__main__":
    print("=== Kiểm thử độc lập Module 6: visualizer.py (TrackingVisualizer) ===")
    vis = TrackingVisualizer(yaw_deadband_px=15.0)
    vis.update_fps(now=0.0)
    fps = vis.update_fps(now=0.033)
    p1 = DetectedPerson(track_id=1, bbox=(360, 120, 440, 300), conf=0.95)
    stats = TelemetryStats(
        fps=fps,
        infer_latency_ms=17.5,
        total_latency_ms=20.0,
        locked_id=1,
        num_targets=1,
        error_x_px=80.0,
        ref_height_px=140.0,
        current_height_px=180.0,
        yaw_rate_deg_s=6.4,
        vx_m_s=-0.50,
    )
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    out = vis.draw_hud(canvas, [p1], stats)
    assert out.shape == (480, 640, 3) and out.sum() > 0
    vis.log_to_terminal(stats, now=1.0)
    print("[PASS] Vẽ HUD và xuất log Terminal thành công!")
    print("=== Module visualizer.py hoạt động tốt! ===")
