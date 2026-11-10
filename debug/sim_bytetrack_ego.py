#!/usr/bin/env python3
"""
[DEBUG] Mô phỏng vòng kín (offline, KHÔNG cần Gazebo/SITL) để tái hiện lỗi mất ID khi tăng Kp yaw.

Dùng ĐÚNG BYTETracker của ultralytics + PIDController của dự án:
  người đi ngang & đổi hướng  ->  ảnh (camera gắn cứng, HFOV 80°, 640px)  ->  detection (nhiễu, conf)
  -> lọc conf (track_conf) -> BYTETracker -> PID yaw -> trễ pipeline + động học yaw ArduPilot -> drone quay

Ví dụ:
  python3 debug/sim_bytetrack_ego.py                       # quét Kp mặc định
  python3 debug/sim_bytetrack_ego.py --kp 0.08 0.3 1.0 --fps 10 --seeds 10
  python3 debug/sim_bytetrack_ego.py --kp 1.0 --track-conf 0.1   # thử hạ ngưỡng conf vào ByteTrack
  python3 debug/sim_bytetrack_ego.py --kp 1.0 --trace            # in từng frame quanh lúc mất ID
"""
import argparse
import math
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from controller import PIDController  # noqa: E402

from ultralytics.engine.results import Boxes  # noqa: E402
from ultralytics.trackers.byte_tracker import BYTETracker  # noqa: E402

W, H = 640, 480


def make_tracker(match_thresh=0.8, track_buffer=30, fuse_score=True):
    args = SimpleNamespace(
        tracker_type="bytetrack", track_high_thresh=0.25, track_low_thresh=0.1,
        new_track_thresh=0.25, track_buffer=track_buffer, match_thresh=match_thresh,
        fuse_score=fuse_score,
    )
    return BYTETracker(args)


def run_once(kp, a, seed, trace=False):
    rng = np.random.default_rng(seed)
    dt = 1.0 / a.fps
    px_per_deg = W / a.hfov
    pid = PIDController(kp=kp, kd=a.kd, deadband=a.deadband, output_limits=(-a.max_yaw, a.max_yaw))
    tracker = make_tracker(a.match_thresh)

    psi = 0.0            # yaw drone (deg)
    r = 0.0              # tốc độ yaw thực (deg/s)
    xp, D = 0.0, a.dist  # vị trí ngang người (m), khoảng cách (m)
    v_dir = 1.0
    t_next_turn = a.turn_every
    cmd_queue = [0.0] * max(0, a.latency_frames)
    turn_t = -10.0

    locked = None
    switches = lost_frames = flips = sat = 0
    prev_sign = 0
    errs = []
    lost_streak = 0
    trace_buf = []

    n = int(a.duration / dt)
    for k in range(n):
        tt = k * dt
        # --- Người di chuyển, đổi hướng định kỳ (có ngẫu nhiên) ---
        if tt >= t_next_turn:
            v_dir = -v_dir
            turn_t = tt
            t_next_turn = tt + a.turn_every * rng.uniform(0.6, 1.4)
        xp += v_dir * a.speed * dt
        bearing = math.degrees(math.atan2(xp, D))

        # --- Ảnh: camera gắn cứng vào thân -> vị trí ngang phụ thuộc (bearing - yaw) ---
        rel = bearing - psi
        cx = W / 2 + rel * px_per_deg + rng.normal(0, a.noise_px)
        bw = a.box_w * (6.0 / D)
        bh = a.box_h * (6.0 / D)
        cy = H / 2 + rng.normal(0, a.noise_px)
        conf = a.base_conf + rng.normal(0, 0.05)
        if tt - turn_t < a.turn_conf_dur:          # người xoay ngang khi đổi hướng -> conf giảm
            conf = a.turn_conf + rng.normal(0, 0.05)
        conf -= a.blur_k * abs(r)                  # (tuỳ chọn) motion blur khi drone quay
        conf = float(np.clip(conf, 0.0, 0.99))
        x1, x2 = cx - bw / 2, cx + bw / 2
        visible = x2 > 0 and x1 < W and rng.random() > a.miss_prob

        dets = np.zeros((0, 6), dtype=np.float32)
        if visible and conf >= a.yolo_conf_floor:
            x1c, x2c = max(0.0, x1), min(W - 1.0, x2)
            dets = np.array([[x1c, cy - bh / 2, x2c, cy + bh / 2, conf, 0]], dtype=np.float32)
        dets = dets[dets[:, 4] >= a.track_conf]     # <-- giống model.track(conf=...)
        out = tracker.update(Boxes(dets, (H, W)))

        target = None
        if len(out):
            tid = int(out[0, 4])
            if locked is None:
                locked = tid
            if tid != locked:
                switches += 1
                if trace:
                    print(f"   !!! ID SWITCH t={tt:.2f}s #{locked} -> #{tid}")
                    for line in trace_buf[-8:]:
                        print("      " + line)
                locked = tid   # sim: khoá lại ngay để đếm tiếp
            target = out[0, :4]

        # --- Điều khiển (giống YawTracker: mất mục tiêu -> hover, reset PID) ---
        if target is not None:
            err = (target[0] + target[2]) / 2 - W / 2
            yaw_cmd = pid.compute(err, now=tt)
            errs.append(abs(err))
            lost_streak = 0
        else:
            pid.reset()
            yaw_cmd = 0.0
            lost_frames += 1
            lost_streak += 1
        s = int(np.sign(yaw_cmd))
        if s != 0 and prev_sign != 0 and s != prev_sign:
            flips += 1
        if s != 0:
            prev_sign = s
        sat += int(target is not None and pid.last_saturated)

        trace_buf.append(f"t={tt:5.2f} rel={rel:+6.1f}° cx={cx:6.1f} conf={conf:.2f} "
                         f"cmd={yaw_cmd:+6.1f} r={r:+6.1f}°/s ego={-r * dt * px_per_deg:+6.1f}px/f "
                         f"det={'Y' if len(dets) else '-'} trk={'Y' if target is not None else '-'}")

        # --- Trễ pipeline + động học yaw ArduPilot (bậc 1 + giới hạn gia tốc) ---
        cmd_queue.append(yaw_cmd)
        cmd_applied = cmd_queue.pop(0)
        dr = (cmd_applied - r) * min(1.0, dt / a.tau)
        dr = float(np.clip(dr, -a.yaw_accel * dt, a.yaw_accel * dt))
        r += dr
        psi += r * dt

    return {
        "switches": switches, "lost_pct": lost_frames / n, "flips_s": flips / a.duration,
        "sat_pct": sat / n, "err_p50": float(np.median(errs)) if errs else float("nan"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kp", type=float, nargs="+", default=[0.08, 0.15, 0.3, 0.5, 1.0])
    p.add_argument("--kd", type=float, default=0.01)
    p.add_argument("--deadband", type=float, default=15.0)
    p.add_argument("--max-yaw", type=float, default=25.0)
    p.add_argument("--fps", type=float, default=15.0, help="FPS vòng xử lý YOLO thực tế")
    p.add_argument("--latency-frames", type=int, default=1, help="Trễ pipeline (frame)")
    p.add_argument("--tau", type=float, default=0.15, help="Hằng số thời gian đáp ứng yaw (s)")
    p.add_argument("--yaw-accel", type=float, default=180.0, help="Giới hạn gia tốc yaw (°/s²)")
    p.add_argument("--hfov", type=float, default=80.0)
    p.add_argument("--dist", type=float, default=6.0, help="Khoảng cách tới người (m)")
    p.add_argument("--speed", type=float, default=1.4, help="Tốc độ đi ngang (m/s)")
    p.add_argument("--turn-every", type=float, default=2.5, help="Đổi hướng mỗi ~N giây")
    p.add_argument("--box-w", type=float, default=45.0)
    p.add_argument("--box-h", type=float, default=150.0)
    p.add_argument("--noise-px", type=float, default=2.0)
    p.add_argument("--base-conf", type=float, default=0.8)
    p.add_argument("--turn-conf", type=float, default=0.55, help="conf khi người xoay ngang")
    p.add_argument("--turn-conf-dur", type=float, default=0.4)
    p.add_argument("--blur-k", type=float, default=0.0, help="Giảm conf mỗi °/s quay (motion blur)")
    p.add_argument("--miss-prob", type=float, default=0.03)
    p.add_argument("--yolo-conf-floor", type=float, default=0.1)
    p.add_argument("--track-conf", type=float, nargs="+", default=[0.5])
    p.add_argument("--match-thresh", type=float, default=0.8)
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--trace", action="store_true", help="In chi tiết quanh mỗi ID switch (seed 0)")
    a = p.parse_args()

    print(f"FPS={a.fps} trễ={a.latency_frames}f tau={a.tau}s HFOV={a.hfov}° "
          f"({W / a.hfov:.1f}px/°) bbox_w≈{a.box_w * 6 / a.dist:.0f}px, người {a.speed}m/s @ {a.dist}m")
    print(f"{'track_conf':>10} {'Kp':>6} | {'IDsw/phút':>9} {'mất %':>7} {'đổi dấu/s':>9} "
          f"{'bão hoà %':>9} {'|err| p50':>9}")
    print("-" * 70)
    for tc in a.track_conf:
        a.track_conf_cur = tc
        for kp in a.kp:
            a2 = SimpleNamespace(**vars(a)); a2.track_conf = tc
            res = [run_once(kp, a2, s, trace=(a.trace and s == 0)) for s in range(a.seeds)]
            m = {k: np.mean([r[k] for r in res]) for k in res[0]}
            print(f"{tc:>10.2f} {kp:>6.2f} | {m['switches'] * 60 / a.duration:>9.2f} "
                  f"{m['lost_pct'] * 100:>6.1f}% {m['flips_s']:>9.2f} {m['sat_pct'] * 100:>8.1f}% "
                  f"{m['err_p50']:>8.1f}px")
    print("\nĐọc kết quả: nếu IDsw/phút tăng vọt theo Kp => giả thuyết 'quay giật làm Kalman ByteTrack "
          "dự đoán sai' được xác nhận.\nThử thêm: --fps 8 (CPU chậm), --track-conf 0.5 0.1, --blur-k 0.01.")


if __name__ == "__main__":
    main()
