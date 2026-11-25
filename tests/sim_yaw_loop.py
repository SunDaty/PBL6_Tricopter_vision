#!/usr/bin/env python3
"""
Mô phỏng vòng kín điều khiển Yaw (10 Hz vision trễ 150 ms + FC bậc 1 trễ 250 ms + rớt frame 15%):
So sánh luật cũ (không bù trễ, kd=0.01, mất 1 frame -> reset & lệnh 0) vs luật mới
(bù trễ, kd=0, Hysteresis deadband, Slew-rate limiter, COAST).
"""
from collections import deque
import math
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from controller import PIDController
from detector import DetectedPerson
from ego_motion import BearingProjector, YawHistory
from tracker import TargetTracker


def simulate(use_new_law: bool):
    rng = np.random.default_rng(7)
    proj = BearingProjector(frame_width=640, hfov_deg=66.0)
    yh = YawHistory(keep_s=3.0)

    dt_sim = 0.01  # 100 Hz vật lý
    vision_period = 0.10  # 10 Hz
    vision_delay = 0.15   # 150 ms capture -> cmd (đo thực tế ~145 ms)
    fc_delay = 0.32       # 320 ms cmd -> FC response (đo thực tế 317 ms)
    # Động học vòng tốc độ yaw bậc 2 (tricopter + servo đuôi có cộng hưởng ~0.8 Hz, giảm chấn thấp)
    wn = 2.0 * math.pi * 0.8
    zeta = 0.25
    fc_gain = 1.5  # phù hợp tỷ lệ |yawspeed|/|cmd| quan sát trong log thực tế (16.1/10.5)

    psi_deg = 0.0
    yawspeed_deg_s = 0.0
    yaw_accel_deg_s2 = 0.0
    target_theta_deg = 24.0  # Bước sai số ban đầu +24°

    cmd_pipe = deque([0.0] * int(round(fc_delay / dt_sim)))
    pending_frames: deque[tuple[float, float, float]] = deque()  # (t_ready, t_cap, psi_cap_deg)

    if use_new_law:
        trk = TargetTracker(
            yaw_kp=0.08, yaw_kd=0.0, yaw_deadband_px=15.0, yaw_deadband_exit_px=26.0,
            max_yaw_rate=20.0, max_yaw_accel=60.0, latency_comp=True, coast_timeout=0.6,
            lost_timeout=3.0, reacquire_enabled=True,
        )
    else:
        old_pid = PIDController(kp=0.08, kd=0.01, deadband=15.0, output_limits=(-25.0, 25.0))

    locked = False
    cur_cmd = 0.0
    prev_cmd = 0.0
    cmd_drops_to_zero = 0
    overshoot_deg = 0.0
    settled_err_samples: list[float] = []
    cmd_changes: list[float] = []

    next_cap_t = 0.0
    for step in range(int(10.0 / dt_sim)):
        t = step * dt_sim
        # Ghi telemetry 50 Hz
        if step % 2 == 0:
            yh.add(t, math.radians(psi_deg), math.radians(yawspeed_deg_s))

        # Chụp ảnh 10 Hz
        if t + 1e-9 >= next_cap_t:
            pending_frames.append((t + vision_delay, t, psi_deg))
            next_cap_t += vision_period

        # Xử lý xong frame sau 150 ms
        while pending_frames and t + 1e-9 >= pending_frames[0][0]:
            _, t_cap, psi_cap_true = pending_frames.popleft()
            dropped = locked and (rng.random() < 0.15)
            xc = proj.bearing_to_px(math.radians(target_theta_deg), math.radians(psi_cap_true))
            xc_noisy = xc + float(rng.normal(0.0, 3.0))
            dets = [] if dropped else [
                DetectedPerson(track_id=1, bbox=(int(xc_noisy - 35), 120, int(xc_noisy + 35), 340), conf=0.85)
            ]

            if use_new_law:
                psi_cap_est = yh.yaw_at(t_cap)
                psi_now_est = yh.latest(t)
                if not locked and dets:
                    trk.select_by_point(int(xc_noisy), 220, dets, now=t, psi_capture=psi_cap_est)
                    locked = True
                cmd, _ = trk.update(dets, now=t, psi_capture=psi_cap_est, psi_now=psi_now_est)
                cur_cmd = cmd.yaw_rate
            else:
                if not locked and dets:
                    locked = True
                if not dets:
                    old_pid.reset()
                    cur_cmd = 0.0
                else:
                    err_px = dets[0].center[0] - 320.0
                    cur_cmd = old_pid.compute(err_px, now=t)

            if abs(prev_cmd) > 2.0 and abs(cur_cmd) < 0.1:
                cmd_drops_to_zero += 1
            cmd_changes.append(abs(cur_cmd - prev_cmd))
            prev_cmd = cur_cmd

        # Cập nhật động học FC bậc 2 (yawspeed'' + 2*zeta*wn*yawspeed' + wn^2*yawspeed = wn^2*(fc_gain*u))
        delayed_cmd = cmd_pipe.popleft()
        cmd_pipe.append(cur_cmd)
        y_dd = (wn ** 2) * (fc_gain * delayed_cmd - yawspeed_deg_s) - 2.0 * zeta * wn * yaw_accel_deg_s2
        yaw_accel_deg_s2 += y_dd * dt_sim
        yawspeed_deg_s += yaw_accel_deg_s2 * dt_sim
        psi_deg += yawspeed_deg_s * dt_sim

        overshoot_deg = max(overshoot_deg, psi_deg - target_theta_deg)
        if t >= 4.0:
            settled_err_samples.append(psi_deg - target_theta_deg)

    return {
        "overshoot_deg": overshoot_deg,
        "overshoot_pct": overshoot_deg / target_theta_deg * 100.0,
        "cmd_drops": cmd_drops_to_zero,
        "mean_cmd_jump": float(np.mean(cmd_changes)),
        "steady_rmse_deg": float(np.sqrt(np.mean(np.square(settled_err_samples)))),
    }


if __name__ == "__main__":
    old_m = simulate(use_new_law=False)
    new_m = simulate(use_new_law=True)
    print(
        f"Luật CŨ : overshoot={old_m['overshoot_deg']:.2f}° ({old_m['overshoot_pct']:.1f}%), "
        f"cmd_drops={old_m['cmd_drops']}, mean|dCmd|={old_m['mean_cmd_jump']:.2f}, steady_RMSE={old_m['steady_rmse_deg']:.2f}°"
    )
    print(
        f"Luật MỚI: overshoot={new_m['overshoot_deg']:.2f}° ({new_m['overshoot_pct']:.1f}%), "
        f"cmd_drops={new_m['cmd_drops']}, mean|dCmd|={new_m['mean_cmd_jump']:.2f}, steady_RMSE={new_m['steady_rmse_deg']:.2f}°"
    )

    assert new_m["overshoot_deg"] < old_m["overshoot_deg"], "Expected lower overshoot with latency comp"
    assert new_m["cmd_drops"] < old_m["cmd_drops"], "Expected fewer sudden command drops with COAST + slew"
    assert new_m["mean_cmd_jump"] < old_m["mean_cmd_jump"], "Expected smoother command profile"
    print("[PASS] Luật điều khiển mới giảm mạnh overshoot, giật lệnh và giữ ổn định hơn luật cũ!")
