# src/tracker.py
from collections import deque
import math
import time
from typing import Optional
import cv2
import numpy as np

from controller import ControlCommand, HysteresisDeadband, PIDController, SlewRateLimiter
from detector import DetectedPerson
from ego_motion import BearingProjector, wrap_angle_rad


def _extract_torso_hist(
    frame: Optional[np.ndarray],
    bbox: tuple[int, int, int, int],
) -> Optional[np.ndarray]:
    """Trích xuất histogram HSV (H 16 x S 16) vùng thân người để nhận lại khi đổi ID."""
    if frame is None or frame.ndim != 3:
        return None
    img_h, img_w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if w < 12 or h < 20:
        return None

    rx1 = max(0, min(img_w - 1, int(x1 + 0.20 * w)))
    rx2 = max(rx1 + 1, min(img_w, int(x1 + 0.80 * w)))
    ry1 = max(0, min(img_h - 1, int(y1 + 0.20 * h)))
    ry2 = max(ry1 + 1, min(img_h, int(y1 + 0.75 * h)))
    if (rx2 - rx1) < 6 or (ry2 - ry1) < 8:
        return None

    roi = frame[ry1:ry2, rx1:rx2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
    return hist


class TargetTracker:
    """
    Module điều khiển bám mục tiêu hợp nhất (dùng chung cho Gazebo Sim & Raspberry Pi 5):
    1. Chọn và khóa ID mục tiêu khi người dùng Click chuột trái (`select_by_point`).
    2. Máy trạng thái bám bền vững: IDLE -> TRACK -> COAST -> SEARCH_HOLD -> IDLE,
       kèm tự động bắt lại mục tiêu (RE-ACQUIRE) khi ByteTrack đổi ID (dựa trên góc phương vị
       thế giới, kích thước bbox, histogram màu áo HSV và fallback an toàn khi chỉ có 1 người).
    3. Điều khiển góc quay Yaw (`yaw_rate` deg/s) có bù độ trễ bằng telemetry yaw hiện tại,
       vùng chết có độ trễ (Hysteresis Deadband) và giới hạn gia tốc góc (Slew-Rate Limiter).
    4. Điều khiển khoảng cách Tiến/Lùi (`vx` m/s) độ trễ thấp & mượt (Median-3 + EMA + Slew).
    """
    def __init__(
        self,
        frame_width: int = 640,
        frame_height: int = 480,
        yaw_kp: float = 0.08,
        yaw_kd: float = 0.0,
        yaw_deadband_px: float = 15.0,
        max_yaw_rate: float = 20.0,
        dist_kp: float = 1.6,
        dist_deadband_ratio: float = 0.05,
        min_vx: float = 0.15,
        max_vx: float = 1.2,
        max_vx_accel: float = 1.8,
        max_vx_decel: float = 2.5,
        height_ema_alpha: float = 0.55,
        lost_timeout: float = 3.0,
        auto_lock: bool = False,
        # Các tham số mở rộng (mặc định tương thích ngược hoàn toàn với cấu hình cũ)
        hfov_deg: float = 66.0,
        yaw_deadband_exit_px: Optional[float] = None,
        max_yaw_accel: float = 60.0,
        latency_comp: bool = True,
        coast_timeout: float = 0.6,
        bearing_ema_alpha: float = 0.55,
        reacquire_enabled: bool = True,
        reacquire_max_bearing_deg: float = 12.0,
        reacquire_height_ratio: tuple[float, float] = (0.65, 1.5),
        reacquire_use_color_hist: bool = True,
        reacquire_hist_min_corr: float = 0.45,
        single_person_fallback: bool = True,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.center_x = frame_width / 2.0
        self.yaw_kp = float(yaw_kp)
        self.yaw_kd = float(yaw_kd)
        self.yaw_deadband_px = float(yaw_deadband_px)
        self.max_yaw_rate = float(max_yaw_rate)
        if yaw_deadband_exit_px is None:
            self.yaw_deadband_exit_px = (
                max(self.yaw_deadband_px * 1.6, self.yaw_deadband_px + 8.0)
                if self.yaw_deadband_px > 0.0
                else 0.0
            )
        else:
            self.yaw_deadband_exit_px = max(self.yaw_deadband_px, float(yaw_deadband_exit_px))

        self.dist_kp = dist_kp
        self.dist_deadband_ratio = dist_deadband_ratio
        self.min_vx = min_vx
        self.max_vx = max_vx
        self.max_vx_accel = max_vx_accel
        self.max_vx_decel = max_vx_decel
        self.height_ema_alpha = height_ema_alpha
        self.lost_timeout = float(lost_timeout)
        self.auto_lock = bool(auto_lock)

        self.latency_comp = bool(latency_comp)
        self.coast_timeout = max(0.0, float(coast_timeout))
        self.bearing_ema_alpha = max(0.1, min(1.0, float(bearing_ema_alpha)))
        self.reacquire_enabled = bool(reacquire_enabled)
        self.reacquire_max_bearing_deg = float(reacquire_max_bearing_deg)
        self.reacquire_height_min = float(reacquire_height_ratio[0])
        self.reacquire_height_max = float(reacquire_height_ratio[1])
        self.reacquire_use_color_hist = bool(reacquire_use_color_hist)
        self.reacquire_hist_min_corr = float(reacquire_hist_min_corr)
        self.single_person_fallback = bool(single_person_fallback)

        self.projector = BearingProjector(frame_width=frame_width, hfov_deg=hfov_deg)

        # Trạng thái mục tiêu đang khóa
        self.state: str = "IDLE"  # IDLE | TRACK | COAST | SEARCH_HOLD
        self.locked_id: Optional[int] = None
        self.last_seen_time: Optional[float] = None
        self.ref_height_px: Optional[float] = None
        self._ref_width_px: Optional[float] = None
        self.current_height_px: Optional[float] = None
        self._last_height_px: Optional[float] = None
        self._height_window: deque[float] = deque(maxlen=3)
        self._current_vx: float = 0.0
        self._last_vx_time: Optional[float] = None

        # Trạng thái góc phương vị thế giới & Re-Acquire
        self._theta_target: Optional[float] = None
        self._omega_person: float = 0.0
        self._last_theta_time: Optional[float] = None
        self._coexisting_ids: set[int] = set()
        self._appearance_hist: Optional[np.ndarray] = None

        # Thông số phơi ra cho HUD / metrics / log sự kiện
        self.reacquire_count: int = 0
        self.last_reacquired: Optional[tuple[int, int]] = None  # (old_id, new_id) trong frame hiện tại
        self.last_lost_id: Optional[int] = None                # ID vừa bị hủy khóa do quá lost_timeout
        self.last_error_deg_now: Optional[float] = None
        self.predicted_x_px: Optional[float] = None

        # Bộ điều khiển Yaw: Hysteresis Deadband + PD (mặc định kd=0) + Slew-Rate Limiter
        self.yaw_deadband = HysteresisDeadband(
            enter=self.yaw_deadband_px,
            exit=self.yaw_deadband_exit_px,
        )
        self.yaw_pid = PIDController(
            kp=self.yaw_kp,
            kd=self.yaw_kd,
            deadband=0.0,  # Deadband đã xử lý bởi HysteresisDeadband
            output_limits=(-self.max_yaw_rate, self.max_yaw_rate),
        )
        self.yaw_slew = SlewRateLimiter(max_rate=max_yaw_accel)

    def reset(self):
        """Xóa mục tiêu đang khóa và đưa bộ điều khiển về trạng thái chờ (Hover)."""
        self.state = "IDLE"
        self.locked_id = None
        self.last_seen_time = None
        self.ref_height_px = None
        self._ref_width_px = None
        self.current_height_px = None
        self._last_height_px = None
        self._height_window.clear()
        self._current_vx = 0.0
        self._last_vx_time = None
        self._theta_target = None
        self._omega_person = 0.0
        self._last_theta_time = None
        self._coexisting_ids.clear()
        self._appearance_hist = None
        self.last_error_deg_now = None
        self.predicted_x_px = None
        self.yaw_deadband.reset()
        self.yaw_pid.reset()
        self.yaw_slew.reset(0.0)

    def _calibrate_reference(
        self,
        person: DetectedPerson,
        now: float,
        frame: Optional[np.ndarray] = None,
        psi_capture: Optional[float] = None,
        detections: Optional[list[DetectedPerson]] = None,
    ):
        """Chốt chiều cao, bề rộng, góc phương vị và mẫu màu lúc vừa chọn mục tiêu."""
        raw_h = max(50.0, min(float(self.frame_height * 0.85), person.height))
        self.ref_height_px = raw_h
        self._ref_width_px = max(20.0, person.width)
        self.current_height_px = raw_h
        self._last_height_px = person.height
        self._height_window.clear()
        self._height_window.append(raw_h)
        self._current_vx = 0.0
        self._last_vx_time = now

        psi_c = psi_capture if (self.latency_comp and psi_capture is not None) else 0.0
        self._theta_target = self.projector.px_to_bearing(person.center[0], psi_c)
        self._omega_person = 0.0
        self._last_theta_time = now
        self._coexisting_ids = (
            {d.track_id for d in detections if d.track_id != person.track_id}
            if detections
            else set()
        )
        self._appearance_hist = (
            _extract_torso_hist(frame, person.bbox) if self.reacquire_use_color_hist else None
        )

        self.yaw_deadband.reset()
        self.yaw_pid.reset()
        self.yaw_slew.reset(0.0)

    def select_by_point(
        self,
        x: int,
        y: int,
        detections: list[DetectedPerson],
        now: Optional[float] = None,
        frame: Optional[np.ndarray] = None,
        psi_capture: Optional[float] = None,
    ) -> Optional[int]:
        """
        Chọn mục tiêu tại tọa độ click chuột trái (x, y) và chốt ngay cự ly hiện tại làm chuẩn.
        Nếu nhiều khung bao chồng lấn, ưu tiên người có tâm gần điểm click nhất.
        """
        candidates = [d for d in detections if d.contains_point(x, y)]
        if not candidates:
            return None

        current_time = now if now is not None else time.monotonic()
        chosen = min(
            candidates,
            key=lambda d: (d.center[0] - x) ** 2 + (d.center[1] - y) ** 2,
        )
        self.locked_id = chosen.track_id
        self.last_seen_time = current_time
        self.state = "TRACK"
        self._calibrate_reference(
            chosen, now=current_time, frame=frame, psi_capture=psi_capture, detections=detections
        )
        return chosen.track_id

    def get_occlusion_elapsed(self, now: Optional[float] = None) -> float:
        """Thời gian (giây) kể từ lần cuối nhìn thấy mục tiêu đang khóa."""
        if self.locked_id is None or self.last_seen_time is None:
            return 0.0
        current_time = now if now is not None else time.monotonic()
        return max(0.0, current_time - self.last_seen_time)

    def _update_appearance(self, person: DetectedPerson, frame: Optional[np.ndarray]):
        if not self.reacquire_use_color_hist or frame is None:
            return
        x1, y1, x2, y2 = person.bbox
        margin = 8
        not_clipped = (
            x1 > margin and x2 < self.frame_width - margin
            and y1 > margin and y2 < self.frame_height - margin
        )
        if self._appearance_hist is not None and (person.conf < 0.6 or not not_clipped):
            return
        hist = _extract_torso_hist(frame, person.bbox)
        if hist is None:
            return
        if self._appearance_hist is None:
            self._appearance_hist = hist
        else:
            blended = 0.9 * self._appearance_hist + 0.1 * hist
            cv2.normalize(blended, blended, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
            self._appearance_hist = blended

    def _try_reacquire(
        self,
        detections: list[DetectedPerson],
        elapsed: float,
        frame: Optional[np.ndarray],
        psi_capture: Optional[float],
    ) -> Optional[DetectedPerson]:
        """Tự động nhận lại mục tiêu khi ByteTrack đổi sang ID mới."""
        if not self.reacquire_enabled or not detections or self._theta_target is None:
            return None

        psi_c = psi_capture if (self.latency_comp and psi_capture is not None) else 0.0
        pred_dt = min(elapsed, self.coast_timeout)
        theta_pred = self._theta_target + self._omega_person * pred_dt
        ref_h = max(1.0, self._last_height_px or self.current_height_px or self.ref_height_px or 100.0)
        max_ang_deg = min(28.0, self.reacquire_max_bearing_deg + 4.0 * elapsed)

        best_cand: Optional[DetectedPerson] = None
        best_score = float("inf")

        for cand in detections:
            if cand.track_id in self._coexisting_ids:
                continue

            theta_cand = self.projector.px_to_bearing(cand.center[0], psi_c)
            ang_diff_deg = abs(math.degrees(wrap_angle_rad(theta_cand - theta_pred)))
            if ang_diff_deg > max_ang_deg:
                continue

            h_ratio = cand.height / ref_h
            if h_ratio < self.reacquire_height_min or h_ratio > self.reacquire_height_max:
                continue

            hist_corr = 0.0
            if self.reacquire_use_color_hist and frame is not None and self._appearance_hist is not None:
                cand_hist = _extract_torso_hist(frame, cand.bbox)
                if cand_hist is not None:
                    hist_corr = float(
                        cv2.compareHist(self._appearance_hist, cand_hist, cv2.HISTCMP_CORREL)
                    )
                    if hist_corr < self.reacquire_hist_min_corr:
                        continue

            score = (
                (ang_diff_deg / max(1.0, max_ang_deg))
                + 0.5 * abs(math.log(max(0.1, h_ratio)))
                - 0.3 * hist_corr
            )
            if score < best_score:
                best_score = score
                best_cand = cand

        if best_cand is not None:
            return best_cand

        # Fallback an toàn cho kịch bản chỉ có duy nhất 1 người trong khung hình
        if self.single_person_fallback and len(detections) == 1:
            sole = detections[0]
            if sole.track_id not in self._coexisting_ids:
                theta_sole = self.projector.px_to_bearing(sole.center[0], psi_c)
                ang_diff_deg = abs(math.degrees(wrap_angle_rad(theta_sole - theta_pred)))
                h_ratio = sole.height / ref_h
                if ang_diff_deg <= 22.0 and 0.55 <= h_ratio <= 1.75:
                    return sole

        return None

    def _compute_effective_height(self, person: DetectedPerson) -> float:
        """
        Tính chiều cao mục tiêu có khử nhiễu gai (Median-of-3) + EMA nhanh (alpha=0.55):
        - Khi người tiến quá sát làm chân/đầu chạm mép ảnh (y2 >= H-8 hoặc y1 <= 8)
          và bề rộng phình to hơn lúc khóa, ước lượng qua tỉ lệ bề rộng để drone lùi lại an toàn.
        """
        _, y1, _, y2 = person.bbox
        raw_h = person.height
        raw_w = person.width

        is_clipped = (y2 >= self.frame_height - 8) or (y1 <= 8)
        if is_clipped and self.ref_height_px is not None and self._ref_width_px is not None:
            width_scale = raw_w / self._ref_width_px
            if width_scale > 1.08:
                raw_h = max(raw_h, self.ref_height_px * width_scale)

        self._height_window.append(raw_h)
        sorted_w = sorted(self._height_window)
        median_h = sorted_w[len(sorted_w) // 2]

        if self.current_height_px is None:
            self.current_height_px = median_h
        else:
            self.current_height_px = (
                self.height_ema_alpha * median_h
                + (1.0 - self.height_ema_alpha) * self.current_height_px
            )
        return self.current_height_px

    def _compute_smooth_vx(self, cur_h: float, now: float) -> float:
        """
        Tính vận tốc tiến/lùi vx (m/s) có bù quán tính khởi động và giới hạn gia tốc (Slew-Rate).
        """
        if self.ref_height_px is None or cur_h <= 1.0 or self.max_vx <= 0.0:
            self._current_vx = 0.0
            self._last_vx_time = now
            return 0.0

        # Sai số khoảng cách đối xứng chuẩn quang học:
        # > 0 khi người đi xa (cur_h < ref_h), < 0 khi người lại gần (cur_h > ref_h)
        if cur_h <= self.ref_height_px:
            dist_error = (self.ref_height_px / cur_h) - 1.0
        else:
            dist_error = -((cur_h / self.ref_height_px) - 1.0)

        # Vùng chết mềm ±5% kèm bù vận tốc khởi động tối thiểu (min_vx)
        if abs(dist_error) <= self.dist_deadband_ratio:
            target_vx = 0.0
        else:
            eff_err = math.copysign(abs(dist_error) - self.dist_deadband_ratio, dist_error)
            raw_vx = math.copysign(self.min_vx, eff_err) + self.dist_kp * eff_err
            target_vx = max(-self.max_vx, min(self.max_vx, raw_vx))

        # Giới hạn gia tốc (Slew-Rate Limiter) giúp lệnh vận tốc biến thiên mượt mà
        dt = (
            now - self._last_vx_time
            if (self._last_vx_time is not None and now > self._last_vx_time)
            else 0.0
        )
        self._last_vx_time = now

        if dt <= 0.0 or dt > 0.5:
            self._current_vx = target_vx
            return self._current_vx

        delta_vx = target_vx - self._current_vx
        is_braking = (abs(target_vx) < abs(self._current_vx)) or (target_vx * self._current_vx < 0.0)
        rate_limit = self.max_vx_decel if is_braking else self.max_vx_accel
        max_step = rate_limit * dt

        if abs(delta_vx) > max_step:
            self._current_vx += math.copysign(max_step, delta_vx)
        else:
            self._current_vx = target_vx

        return self._current_vx

    def update(
        self,
        detections: list[DetectedPerson],
        now: Optional[float] = None,
        frame: Optional[np.ndarray] = None,
        psi_capture: Optional[float] = None,
        psi_now: Optional[float] = None,
    ) -> tuple[ControlCommand, Optional[DetectedPerson]]:
        """
        Cập nhật danh sách phát hiện ở frame hiện tại và tính lệnh điều khiển (vx, yaw_rate).
        - Có bù độ trễ bằng (psi_capture, psi_now) nếu latency_comp=True và có telemetry yaw.
        - Khi mất ID: thử RE-ACQUIRE trước, nếu vẫn chưa thấy thì COAST mượt trong `coast_timeout`,
          sau đó giảm về 0 (SEARCH_HOLD) và chỉ hủy khóa sau `lost_timeout`.
        """
        current_time = now if now is not None else time.monotonic()
        self.last_reacquired = None
        self.last_lost_id = None
        self.predicted_x_px = None

        if self.locked_id is None:
            if self.auto_lock and detections:
                chosen = min(detections, key=lambda d: d.track_id)
                self.locked_id = chosen.track_id
                self.last_seen_time = current_time
                self.state = "TRACK"
                self._calibrate_reference(
                    chosen,
                    now=current_time,
                    frame=frame,
                    psi_capture=psi_capture,
                    detections=detections,
                )
            else:
                self.state = "IDLE"
                self.last_error_deg_now = None
                return ControlCommand(), None

        target = next((d for d in detections if d.track_id == self.locked_id), None)

        if target is None:
            elapsed = self.get_occlusion_elapsed(now=current_time)
            # 1. Thử tự bắt lại mục tiêu nếu ByteTrack đổi ID
            reacq = self._try_reacquire(detections, elapsed, frame, psi_capture)
            if reacq is not None:
                old_id = self.locked_id
                self.locked_id = reacq.track_id
                self.reacquire_count += 1
                self.last_reacquired = (old_id, reacq.track_id)
                target = reacq

        if target is None:
            elapsed = self.get_occlusion_elapsed(now=current_time)
            self._current_vx = 0.0
            self._last_vx_time = current_time

            # 2. Quá lost_timeout -> tự động hủy khóa về IDLE
            if elapsed > self.lost_timeout:
                self.last_lost_id = self.locked_id
                self.reset()
                return ControlCommand(), None

            # 3. COAST: giữ lệnh hướng về vị trí dự đoán và suy giảm dần về 0 (không giật on/off)
            if elapsed <= self.coast_timeout and self._theta_target is not None:
                self.state = "COAST"
                theta_pred = self._theta_target + self._omega_person * elapsed
                psi_ref = (
                    psi_now
                    if (self.latency_comp and psi_now is not None)
                    else (psi_capture if (self.latency_comp and psi_capture is not None) else 0.0)
                )
                self.predicted_x_px = self.projector.bearing_to_px(theta_pred, psi_ref)
                err_rad = wrap_angle_rad(theta_pred - psi_ref)
                self.last_error_deg_now = math.degrees(err_rad)
                eq_err_px = self.projector.angle_rad_to_eq_px(err_rad)
                eff_err = self.yaw_deadband.apply(eq_err_px)
                decay = max(0.0, 1.0 - elapsed / max(1e-3, self.coast_timeout))
                raw_yaw = max(-self.max_yaw_rate, min(self.max_yaw_rate, self.yaw_kp * eff_err * decay))
                yaw_rate = self.yaw_slew.step(raw_yaw, now=current_time)
                return ControlCommand(vx=0.0, vy=0.0, vz=0.0, yaw_rate=yaw_rate), None

            # 4. SEARCH_HOLD: quá coast_timeout nhưng chưa quá lost_timeout -> giảm mượt về 0
            self.state = "SEARCH_HOLD"
            self.yaw_pid.reset()
            self.last_error_deg_now = None
            if self._theta_target is not None:
                psi_ref = (
                    psi_now
                    if (self.latency_comp and psi_now is not None)
                    else (psi_capture if (self.latency_comp and psi_capture is not None) else 0.0)
                )
                self.predicted_x_px = self.projector.bearing_to_px(self._theta_target, psi_ref)
            yaw_rate = self.yaw_slew.step(0.0, now=current_time)
            return ControlCommand(vx=0.0, vy=0.0, vz=0.0, yaw_rate=yaw_rate), None

        # --- TRẠNG THÁI TRACK ---
        self.state = "TRACK"
        self.last_seen_time = current_time
        self._last_height_px = target.height
        self._coexisting_ids = {d.track_id for d in detections if d.track_id != target.track_id}

        if self.ref_height_px is None:
            self._calibrate_reference(
                target,
                now=current_time,
                frame=frame,
                psi_capture=psi_capture,
                detections=detections,
            )
        else:
            self._update_appearance(target, frame)

        # Cập nhật góc phương vị thế giới của mục tiêu
        has_telem = self.latency_comp and psi_capture is not None
        psi_c = psi_capture if has_telem else 0.0
        theta_meas = self.projector.px_to_bearing(target.center[0], psi_c)
        if self._theta_target is None or self._last_theta_time is None:
            self._theta_target = theta_meas
            self._omega_person = 0.0
        else:
            d_theta = wrap_angle_rad(theta_meas - self._theta_target)
            dt_obs = current_time - self._last_theta_time
            if 0.02 <= dt_obs <= 1.0:
                max_omega = math.radians(30.0)
                omega_inst = max(-max_omega, min(max_omega, d_theta / dt_obs))
                self._omega_person = 0.3 * omega_inst + 0.7 * self._omega_person
            self._theta_target = self._theta_target + self.bearing_ema_alpha * d_theta
        self._last_theta_time = current_time

        # 1. Tính tốc độ góc quay Yaw (deg/s) có bù độ trễ + Hysteresis + Slew-Rate Limiter
        if has_telem and psi_now is not None:
            err_rad = wrap_angle_rad(self._theta_target - psi_now)
            eq_err_px = self.projector.angle_rad_to_eq_px(err_rad)
            self.last_error_deg_now = math.degrees(err_rad)
        else:
            eq_err_px = target.center[0] - self.center_x
            self.last_error_deg_now = math.degrees(math.atan(eq_err_px / self.projector.focal_px))

        eff_err = self.yaw_deadband.apply(eq_err_px)
        if eff_err == 0.0:
            self.yaw_pid.reset()
            raw_yaw = 0.0
        else:
            raw_yaw = self.yaw_pid.compute(eff_err, now=current_time)
        yaw_rate = self.yaw_slew.step(raw_yaw, now=current_time)

        # 2. Tính vận tốc Tiến/Lùi Vx (m/s) mượt và độ trễ thấp
        cur_h = self._compute_effective_height(target)
        vx = self._compute_smooth_vx(cur_h, now=current_time)

        return ControlCommand(vx=vx, vy=0.0, vz=0.0, yaw_rate=yaw_rate), target


# Alias tương thích
CombinedTracker = TargetTracker


if __name__ == "__main__":
    print("=== Kiểm thử độc lập Module 5: tracker.py (TargetTracker: Select + Yaw + Smooth Distance) ===")
    tracker = TargetTracker(height_ema_alpha=1.0, lost_timeout=2.0)

    # 1. Chưa chọn mục tiêu -> Drone Hover đứng yên (vx=0, yaw_rate=0)
    p1 = DetectedPerson(track_id=1, bbox=(400, 170, 480, 310), conf=0.92)
    cmd_idle, active = tracker.update([p1], now=0.0)
    assert active is None and cmd_idle.vx == 0.0 and cmd_idle.yaw_rate == 0.0
    print("[PASS] 1. Khi chưa click chọn mục tiêu -> Drone Hover đứng yên")

    # 2. Click chuột trái vào người #1 (cao 140px, tâm x=440 lệch phải +120px)
    chosen_id = tracker.select_by_point(440, 240, [p1], now=0.1)
    cmd_lock, active = tracker.update([p1], now=0.1)
    assert chosen_id == 1 and active is not None
    assert cmd_lock.vx == 0.0 and cmd_lock.yaw_rate > 0.0
    print(
        f"[PASS] 2. Click khóa ID #{chosen_id} (ref_h={tracker.ref_height_px:.0f}px): "
        f"yaw_rate={cmd_lock.yaw_rate:+.2f} deg/s, vx={cmd_lock.vx:+.2f} m/s (Không bị tự trôi)"
    )

    # 3. Người đi ra xa (chiều cao giảm xuống 90px) -> Tăng tốc mượt tiến theo (vx > 0)
    p1_far = DetectedPerson(track_id=1, bbox=(400, 195, 480, 285), conf=0.92)
    tracker.update([p1_far], now=0.3)
    cmd_far, _ = tracker.update([p1_far], now=0.5)
    assert cmd_far.vx > 0.0
    print(f"[PASS] 3. Mục tiêu đi xa (h={tracker.current_height_px:.0f}px) -> vx={cmd_far.vx:+.2f} m/s (Tiến theo mượt)")

    # 4. Người bước lại gần (chiều cao tăng lên 200px) -> Đảo chiều mượt lùi lại (vx < 0)
    p1_close = DetectedPerson(track_id=1, bbox=(380, 140, 500, 340), conf=0.95)
    tracker.update([p1_close], now=0.8)
    cmd_close, _ = tracker.update([p1_close], now=1.2)
    assert cmd_close.vx < 0.0
    print(f"[PASS] 4. Mục tiêu lại gần (h={tracker.current_height_px:.0f}px) -> vx={cmd_close.vx:+.2f} m/s (Lùi lại mượt)")

    # 5. Mất dấu quá lost_timeout (2.0s) -> Tự động về trạng thái chờ chọn lại
    tracker.update([], now=3.5)
    assert tracker.locked_id is None
    print("[PASS] 5. Khuất bóng quá 2.0s -> Tự động hủy khóa và Hover an toàn")
    print("=== Module tracker.py hoạt động tốt! ===")
