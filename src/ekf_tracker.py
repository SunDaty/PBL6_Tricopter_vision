# src/ekf_tracker.py
"""
Bộ lọc Kalman Mở rộng (Extended Kalman Filter - EKF) & Dự đoán Quỹ đạo (Trajectory Predictor)
cho bài toán Drone bám mục tiêu sử dụng mô hình Camera Pinhole phi tuyến có bù chuyển động quay
của thân drone (Ego-motion Compensation).

1. Tại sao dùng EKF (Extended Kalman Filter)?
   - Khi camera gắn cứng trên drone quay với tốc độ góc yaw_rate (ω_z), tọa độ ngang u của mục tiêu
     trên mặt phẳng ảnh biến thiên PHI TUYẾN theo mô hình chiếu phối cảnh (Pinhole Camera Model):
         θ = atan((u - cx0) / fx)
         u_ego = cx0 + fx * tan(θ - ω_z * dt)
   - EKF tuyến tính hóa hàm chuyển tiếp trạng thái phi tuyến f(x, u, dt) quanh ước lượng hiện tại
     thông qua ma trận Jacobian F = ∂f/∂x.

2. Vector trạng thái 8 chiều (State Vector):
   x = [u, v, bw, bh, vu, vv, au, av]^T
   Trong đó:
     - (u, v)   : Tọa độ tâm bounding box trên ảnh (pixel)
     - (bw, bh) : Chiều rộng và chiều cao bounding box (pixel)
     - (vu, vv) : Vận tốc thực của mục tiêu trên ảnh ĐÃ TRỪ chuyển động quay của drone (px/s)
     - (au, av) : Gia tốc của mục tiêu trên ảnh (px/s^2)

3. Dự đoán Quỹ đạo (Trajectory Prediction):
   - Ngoại suy chuỗi điểm tương lai (Horizon Prediction) phục vụ hiển thị HUD và đón đầu mục tiêu.
   - Chế độ trượt quán tính (Coasting) khi YOLO mất bbox tạm thời.
   - Bù độ trễ suy luận (Latency Look-ahead) cho YOLOv8n PyTorch trên Gazebo.
"""

from dataclasses import dataclass
import math
from typing import Optional
import numpy as np


@dataclass
class EKFPrediction:
    """Kết quả ước lượng từ EKF và dự đoán quỹ đạo tại mỗi frame."""
    bbox: tuple[int, int, int, int]              # Bbox đã lọc nhiễu (hoặc dự đoán khi mất dấu)
    center: tuple[float, float]                  # Tâm (u, v) ước lượng hiện tại
    lookahead_center: tuple[float, float]        # Tâm đón đầu (đã bù độ trễ suy luận AI)
    velocity_px_s: tuple[float, float]           # Vận tốc thực của người (vu, vv) (px/s)
    acceleration_px_s2: tuple[float, float]      # Gia tốc của người (au, av) (px/s^2)
    trajectory_points: list[tuple[int, int]]     # Danh sách các điểm dự đoán quỹ đạo trong tương lai
    is_coasting: bool                            # True nếu đang dùng quỹ đạo dự đoán do mất bbox
    coast_elapsed_s: float                       # Thời gian (s) đang bay theo dự đoán
    position_uncertainty_px: float               # Độ bất định vị trí (căn bậc 2 trace hiệp phương sai)


class TargetEKF:
    """
    Bộ lọc EKF kết hợp dự đoán quỹ đạo đa bước cho mục tiêu đang khóa.
    """
    def __init__(
        self,
        frame_width: int = 640,
        frame_height: int = 480,
        hfov_deg: float = 80.0,
        horizon_s: float = 1.0,
        horizon_steps: int = 10,
        max_coast_s: float = 1.0,
        process_noise_pos: float = 4.0,
        process_noise_vel: float = 25.0,
        process_noise_acc: float = 80.0,
        meas_noise_px: float = 6.0,
        meas_noise_size: float = 10.0,
        vel_damping: float = 0.92,
        acc_damping: float = 0.80,
    ):
        self.frame_width = float(frame_width)
        self.frame_height = float(frame_height)
        self.cx0 = frame_width / 2.0
        self.cy0 = frame_height / 2.0

        # Tiêu cự quy đổi theo pixel từ góc mở ngang HFOV của camera Gazebo:
        # fx = (W / 2) / tan(HFOV / 2)
        half_fov_rad = math.radians(max(10.0, min(160.0, hfov_deg)) / 2.0)
        self.fx = (frame_width / 2.0) / math.tan(half_fov_rad)
        self.fy = self.fx

        self.horizon_s = horizon_s
        self.horizon_steps = max(2, horizon_steps)
        self.max_coast_s = max_coast_s
        self.vel_damping = vel_damping
        self.acc_damping = acc_damping

        # Trọng số nhiễu quá trình (Q) và nhiễu đo lường (R)
        self.q_pos = process_noise_pos
        self.q_vel = process_noise_vel
        self.q_acc = process_noise_acc
        self.r_pos = meas_noise_px
        self.r_size = meas_noise_size

        # Vector trạng thái x (8x1): [u, v, bw, bh, vu, vv, au, av]^T
        self.x: Optional[np.ndarray] = None
        # Ma trận hiệp phương sai P (8x8)
        self.P: Optional[np.ndarray] = None

        self._last_time: Optional[float] = None
        self._last_update_time: Optional[float] = None
        self._initialized: bool = False

        # Ma trận đo lường H (4x8): đo trực tiếp [u, v, bw, bh] từ YOLO bbox
        self.H = np.zeros((4, 8), dtype=float)
        self.H[0, 0] = 1.0  # u
        self.H[1, 1] = 1.0  # v
        self.H[2, 2] = 1.0  # bw
        self.H[3, 3] = 1.0  # bh

        # Ma trận hiệp phương sai nhiễu đo lường cơ sở R (4x4)
        self.R_base = np.diag([
            self.r_pos ** 2,
            self.r_pos ** 2,
            self.r_size ** 2,
            self.r_size ** 2,
        ])

    @property
    def is_initialized(self) -> bool:
        return self._initialized and self.x is not None and self.P is not None

    def reset(self):
        """Xóa trạng thái bộ lọc EKF khi đổi mục tiêu hoặc hủy khóa."""
        self.x = None
        self.P = None
        self._last_time = None
        self._last_update_time = None
        self._initialized = False

    def initialize(self, bbox: tuple[int, int, int, int], now: float):
        """Khởi tạo trạng thái EKF từ bounding box đầu tiên của mục tiêu."""
        x1, y1, x2, y2 = bbox
        u = (x1 + x2) / 2.0
        v = (y1 + y2) / 2.0
        bw = max(4.0, float(x2 - x1))
        bh = max(4.0, float(y2 - y1))

        self.x = np.array([u, v, bw, bh, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.P = np.diag([
            25.0,    # u
            25.0,    # v
            36.0,    # bw
            36.0,    # bh
            400.0,   # vu
            400.0,   # vv
            900.0,   # au
            900.0,   # av
        ])
        self._last_time = now
        self._last_update_time = now
        self._initialized = True

    def _nonlinear_transition(
        self,
        state: np.ndarray,
        dt: float,
        yaw_rate_rad_s: float = 0.0,
        pitch_rate_rad_s: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Hàm chuyển tiếp trạng thái phi tuyến f(x, u_ctrl, dt) và ma trận Jacobian F = ∂f/∂x.
        Mô hình hóa chính xác sự dịch chuyển ảnh Pinhole khi camera quay góc dpsi = yaw_rate * dt.
        """
        u, v, bw, bh, vu, vv, au, av = state

        # 1. Phép chiếu Pinhole phi tuyến cho tọa độ ngang u khi drone quay góc dpsi
        dpsi = yaw_rate_rad_s * dt
        theta_u = math.atan2(u - self.cx0, self.fx)
        theta_u_new = theta_u - dpsi
        # Giới hạn góc trong (-82°, +82°) để tránh kỳ dị tan(±90°)
        theta_u_new = max(-1.43, min(1.43, theta_u_new))
        u_ego = self.cx0 + self.fx * math.tan(theta_u_new)

        # Đạo hàm riêng ∂u_ego/∂u = sec^2(theta_u - dpsi) / sec^2(theta_u)
        cos_old = max(1e-4, math.cos(theta_u))
        cos_new = max(1e-4, math.cos(theta_u_new))
        du_ego_du = (cos_old * cos_old) / (cos_new * cos_new)

        # 2. Phép chiếu Pinhole phi tuyến cho tọa độ dọc v khi drone chúc/ngóc (pitch_rate)
        dtheta = pitch_rate_rad_s * dt
        theta_v = math.atan2(v - self.cy0, self.fy)
        theta_v_new = max(-1.43, min(1.43, theta_v - dtheta))
        v_ego = self.cy0 + self.fy * math.tan(theta_v_new)

        cos_v_old = max(1e-4, math.cos(theta_v))
        cos_v_new = max(1e-4, math.cos(theta_v_new))
        dv_ego_dv = (cos_v_old * cos_v_old) / (cos_v_new * cos_v_new)

        # 3. Động học vị trí - vận tốc - gia tốc (Constant Acceleration có suy giảm mượt)
        half_dt2 = 0.5 * dt * dt
        u_pred = u_ego + vu * dt + au * half_dt2
        v_pred = v_ego + vv * dt + av * half_dt2
        bw_pred = max(8.0, bw)
        bh_pred = max(8.0, bh)

        # Hệ số suy giảm vận tốc/gia tốc theo thời gian giúp quỹ đạo dự đoán không vọt vô hạn
        vd = self.vel_damping ** (dt * 10.0)
        ad = self.acc_damping ** (dt * 10.0)
        vu_pred = (vu + au * dt) * vd
        vv_pred = (vv + av * dt) * vd
        au_pred = au * ad
        av_pred = av * ad

        next_state = np.array(
            [u_pred, v_pred, bw_pred, bh_pred, vu_pred, vv_pred, au_pred, av_pred],
            dtype=float,
        )

        # 4. Ma trận Jacobian F = ∂f/∂x (8x8)
        F = np.eye(8, dtype=float)
        F[0, 0] = du_ego_du
        F[0, 4] = dt
        F[0, 6] = half_dt2

        F[1, 1] = dv_ego_dv
        F[1, 5] = dt
        F[1, 7] = half_dt2

        F[4, 4] = vd
        F[4, 6] = dt * vd
        F[5, 5] = vd
        F[5, 7] = dt * vd

        F[6, 6] = ad
        F[7, 7] = ad

        return next_state, F

    def _build_process_noise(self, dt: float) -> np.ndarray:
        """Xây dựng ma trận hiệp phương sai nhiễu quá trình Q(dt) (8x8)."""
        dt_clamped = max(1e-3, min(0.5, dt))
        q_p = (self.q_pos ** 2) * dt_clamped
        q_s = (self.q_pos ** 2) * 0.5 * dt_clamped
        q_v = (self.q_vel ** 2) * dt_clamped
        q_a = (self.q_acc ** 2) * dt_clamped
        return np.diag([q_p, q_p, q_s, q_s, q_v, q_v, q_a, q_a])

    def predict(
        self,
        now: float,
        yaw_rate_deg_s: float = 0.0,
        pitch_rate_deg_s: float = 0.0,
    ):
        """
        Bước Dự đoán (EKF Time Update / Prediction Step):
          x_{k|k-1} = f(x_{k-1|k-1}, u_k, dt)
          P_{k|k-1} = F_k * P_{k-1|k-1} * F_k^T + Q_k
        """
        if not self.is_initialized:
            return

        dt = now - self._last_time if (self._last_time is not None and now > self._last_time) else 0.0
        self._last_time = now
        if dt <= 1e-5:
            return

        # Giới hạn dt nếu vòng lặp bị ngắt quãng quá lâu
        dt_eff = min(dt, 0.25)
        yaw_rate_rad = math.radians(yaw_rate_deg_s)
        pitch_rate_rad = math.radians(pitch_rate_deg_s)

        self.x, F = self._nonlinear_transition(self.x, dt_eff, yaw_rate_rad, pitch_rate_rad)
        Q = self._build_process_noise(dt_eff)
        self.P = F @ self.P @ F.T + Q

    def update(
        self,
        bbox: tuple[int, int, int, int],
        conf: float,
        now: float,
    ):
        """
        Bước Cập nhật Đo lường (EKF Measurement Update / Correction Step):
          y_k = z_k - h(x_{k|k-1})
          S_k = H * P_{k|k-1} * H^T + R_k
          K_k = P_{k|k-1} * H^T * S_k^{-1}
          x_{k|k} = x_{k|k-1} + K_k * y_k
          P_{k|k} = (I - K_k * H) * P_{k|k-1}
        """
        if not self.is_initialized:
            self.initialize(bbox, now)
            return

        x1, y1, x2, y2 = bbox
        z = np.array([
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
            max(4.0, float(x2 - x1)),
            max(4.0, float(y2 - y1)),
        ], dtype=float)

        # Điều chỉnh nhiễu đo lường R thích nghi theo độ tự tin (conf) của YOLOv8:
        # Khi ảnh mờ / người xoay ngang (conf thấp) -> tăng R để tin vào mô hình động học EKF hơn.
        conf_scale = 1.0 / max(0.15, min(1.0, conf)) ** 2
        R = self.R_base * conf_scale

        # Sai số đo lường (Innovation / Residual)
        y = z - (self.H @ self.x)

        # Hiệp phương sai Innovation
        S = self.H @ self.P @ self.H.T + R
        # Hệ số Kalman Gain K = P * H^T * S^-1
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Cập nhật trạng thái và hiệp phương sai (dạng Joseph ổn định số học)
        self.x = self.x + K @ y
        I_KH = np.eye(8) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        self._last_update_time = now
        self._last_time = now

    def predict_trajectory(
        self,
        horizon_s: Optional[float] = None,
        steps: Optional[int] = None,
        assumed_yaw_rate_deg_s: float = 0.0,
    ) -> list[tuple[int, int]]:
        """
        Dự đoán chuỗi tọa độ (u, v) của mục tiêu trong tương lai (mặc định 1.0 giây tới).
        """
        if not self.is_initialized:
            return []

        total_t = horizon_s if horizon_s is not None else self.horizon_s
        n_steps = steps if steps is not None else self.horizon_steps
        dt_step = total_t / float(max(1, n_steps))

        sim_state = self.x.copy()
        yaw_rad = math.radians(assumed_yaw_rate_deg_s)
        points: list[tuple[int, int]] = []

        for _ in range(n_steps):
            # Ngoại suy quỹ đạo mục tiêu trên mặt phẳng ảnh (với giả định tốc độ quay giảm dần)
            sim_state, _ = self._nonlinear_transition(sim_state, dt_step, yaw_rad * 0.3, 0.0)
            px = int(np.clip(round(sim_state[0]), 0, self.frame_width - 1))
            py = int(np.clip(round(sim_state[1]), 0, self.frame_height - 1))
            points.append((px, py))

        return points

    def get_estimate(
        self,
        now: float,
        lookahead_s: float = 0.05,
        yaw_rate_deg_s: float = 0.0,
    ) -> Optional[EKFPrediction]:
        """
        Trả về kết quả ước lượng hiện tại, điểm đón đầu bù trễ (lookahead) và quỹ đạo tương lai.
        Nếu mục tiêu bị mất dấu quá `max_coast_s`, trả về None để drone dừng Hover an toàn.
        """
        if not self.is_initialized or self._last_update_time is None:
            return None

        coast_elapsed = max(0.0, now - self._last_update_time)
        is_coasting = coast_elapsed > 1e-4

        if is_coasting and coast_elapsed > self.max_coast_s:
            return None

        u, v, bw, bh, vu, vv, au, av = self.x
        u_clamped = float(np.clip(u, 0.0, self.frame_width - 1.0))
        v_clamped = float(np.clip(v, 0.0, self.frame_height - 1.0))
        half_w = max(4.0, bw / 2.0)
        half_h = max(4.0, bh / 2.0)

        est_bbox = (
            int(round(max(0.0, u_clamped - half_w))),
            int(round(max(0.0, v_clamped - half_h))),
            int(round(min(self.frame_width - 1.0, u_clamped + half_w))),
            int(round(min(self.frame_height - 1.0, v_clamped + half_h))),
        )

        # Điểm ngắm đón đầu (Look-ahead Point) bù cho độ trễ suy luận YOLOv8n PyTorch
        la_dt = max(0.0, min(0.20, lookahead_s))
        u_lookahead = float(np.clip(u + vu * la_dt + 0.5 * au * la_dt * la_dt, 0.0, self.frame_width - 1.0))
        v_lookahead = float(np.clip(v + vv * la_dt + 0.5 * av * la_dt * la_dt, 0.0, self.frame_height - 1.0))

        traj = self.predict_trajectory(assumed_yaw_rate_deg_s=yaw_rate_deg_s)
        pos_std = float(math.sqrt(max(0.0, self.P[0, 0] + self.P[1, 1])))

        return EKFPrediction(
            bbox=est_bbox,
            center=(u_clamped, v_clamped),
            lookahead_center=(u_lookahead, v_lookahead),
            velocity_px_s=(float(vu), float(vv)),
            acceleration_px_s2=(float(au), float(av)),
            trajectory_points=traj,
            is_coasting=is_coasting,
            coast_elapsed_s=coast_elapsed,
            position_uncertainty_px=pos_std,
        )
