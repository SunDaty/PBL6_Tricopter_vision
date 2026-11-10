#!/usr/bin/env python3
"""
[DEBUG] Phân tích log do `main_tracker.py --debug` sinh ra.

    python3 debug/analyze_log.py debug_logs/<timestamp>        # thư mục log
    python3 debug/analyze_log.py debug_logs/<timestamp> --plot  # + vẽ biểu đồ PNG

In ra:
  1. Thống kê vòng điều khiển: FPS, dt, rớt frame, % bão hoà, % deadband, tần suất đổi dấu yaw,
     độ trễ lệnh yaw -> tốc độ quay thực tế (cross-correlation), trượt ảnh do quay (px/frame).
  2. Thống kê độ "khoẻ" của phép ghép ByteTrack: phân bố kf_iou & fused_cost.
  3. Mỗi sự kiện mất ID: nguyên nhân nghi ngờ + 1s dữ liệu trước đó.
  4. Kết luận xếp hạng nguyên nhân.
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter

import numpy as np


def load_rows(path):
    with open(os.path.join(path, "frames.csv"), newline="") as f:
        return list(csv.DictReader(f))


def col(rows, name):
    out = []
    for r in rows:
        v = r.get(name, "")
        out.append(float(v) if v not in ("", None) else np.nan)
    return np.array(out, dtype=float)


def pct(x, q):
    x = x[~np.isnan(x)]
    return float(np.percentile(x, q)) if len(x) else float("nan")


def estimate_lag(t, cmd, meas):
    """Ước lượng độ trễ giữa lệnh yaw và tốc độ quay đo được bằng cross-correlation."""
    m = ~np.isnan(cmd) & ~np.isnan(meas)
    if m.sum() < 50:
        return None, None
    t, cmd, meas = t[m], cmd[m], meas[m]
    dt = np.median(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        return None, None
    tg = np.arange(t[0], t[-1], dt)
    c = np.interp(tg, t, cmd) - np.mean(cmd)
    y = np.interp(tg, t, meas) - np.mean(meas)
    if np.std(c) < 1e-3 or np.std(y) < 1e-3:
        return None, None
    max_lag = int(1.0 / dt)
    best, best_k = -1e9, 0
    for k in range(0, max_lag):
        v = np.corrcoef(c[: len(c) - k], y[k:])[0, 1]
        if v > best:
            best, best_k = v, k
    # Hệ số khuếch đại thực tế (meas/cmd) để kiểm tra ArduPilot có đáp ứng đủ lệnh không
    gain = np.polyfit(c[: len(c) - best_k], y[best_k:], 1)[0] if best_k < len(c) else None
    return best_k * dt, gain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--match-thresh", type=float, default=0.8)
    args = ap.parse_args()

    rows = load_rows(args.log_dir)
    if not rows:
        print("Log trống.")
        return
    events = []
    ev_path = os.path.join(args.log_dir, "events.jsonl")
    if os.path.exists(ev_path):
        with open(ev_path) as f:
            events = [json.loads(l) for l in f if l.strip()]

    t = col(rows, "t"); t = t - t[0]
    dt = col(rows, "loop_dt_ms")
    dropped = col(rows, "dropped_since_last")
    infer = col(rows, "infer_ms")
    found = col(rows, "target_found")
    err = col(rows, "err_x")
    yaw_cmd = col(rows, "yaw_cmd")
    sat = col(rows, "saturated")
    dead = col(rows, "in_deadband")
    att_rate = col(rows, "att_yawrate_dps")
    ego = col(rows, "ego_shift_px")
    tdx = col(rows, "target_dx_px")
    tdxw = col(rows, "target_dx_world_px")
    kf_iou = col(rows, "kf_iou")
    fcost = col(rows, "fused_cost")
    conf = col(rows, "conf")
    bw = col(rows, "bw")
    nlow = col(rows, "n_filtered_lowconf")
    locked = np.array([r["state"] == "LOCKED" for r in rows])

    dur = t[-1] if len(t) > 1 else 0
    tracking = locked & (found == 1)
    print("=" * 78)
    print(f" LOG: {args.log_dir}   ({len(rows)} frame, {dur:.1f}s)")
    print("=" * 78)

    # ------------------------------------------------------------------ 1
    print("\n[1] VÒNG ĐIỀU KHIỂN")
    fps = 1000.0 / np.nanmedian(dt[1:]) if len(dt) > 1 else float("nan")
    print(f"  FPS xử lý (median)        : {fps:.1f}  | dt p95={pct(dt[1:], 95):.0f}ms max={np.nanmax(dt):.0f}ms")
    print(f"  YOLO+ByteTrack (median)   : {np.nanmedian(infer):.1f}ms")
    print(f"  Frame Gazebo bị bỏ qua    : {np.nansum(dropped):.0f} "
          f"({np.nansum(dropped) / max(1, np.nansum(dropped) + len(rows)):.0%})")
    if tracking.any():
        print(f"  Khi đang bám:  bão hoà yaw = {np.nanmean(sat[tracking]):.0%} | "
              f"trong deadband = {np.nanmean(dead[tracking]):.0%}")
        s = np.sign(yaw_cmd[tracking]); s = s[s != 0]
        flips = np.sum(s[1:] != s[:-1]) if len(s) > 1 else 0
        tt = np.sum(tracking) / max(fps, 1e-3)
        print(f"  Đổi dấu lệnh yaw          : {flips} lần (~{flips / max(tt, 1e-3):.2f} lần/s)")
        print(f"  |err_x| p50/p95           : {pct(np.abs(err[tracking]), 50):.0f} / "
              f"{pct(np.abs(err[tracking]), 95):.0f} px")
    lag, gain = estimate_lag(t, yaw_cmd, att_rate)
    if lag is not None:
        print(f"  Trễ lệnh yaw -> quay thực : ~{lag * 1000:.0f} ms | gain đo/lệnh ≈ {gain:.2f}")
    else:
        print("  Trễ lệnh yaw -> quay thực : (không đủ dữ liệu ATTITUDE)")
    if not np.all(np.isnan(ego)):
        print(f"  Trượt ảnh do drone quay   : |ego| p50={pct(np.abs(ego), 50):.1f} "
              f"p95={pct(np.abs(ego), 95):.1f} max={np.nanmax(np.abs(ego)):.1f} px/frame")
        if not np.all(np.isnan(bw)):
            ratio = np.abs(ego) / bw
            print(f"  ego / bề rộng bbox (p95)  : {pct(ratio, 95):.0%}  "
                  f"(>30% là nguy hiểm cho phép ghép IoU)")
    if not np.all(np.isnan(tdxw)):
        print(f"  Người tự di chuyển (đã trừ ego) |dx| p95: {pct(np.abs(tdxw), 95):.1f} px/frame")

    # ------------------------------------------------------------------ 2
    print("\n[2] ĐỘ KHOẺ PHÉP GHÉP BYTETRACK (khi đang bám)")
    k = kf_iou[tracking]; k = k[~np.isnan(k)]
    fc = fcost[tracking]; fc = fc[~np.isnan(fc)]
    if len(k):
        print(f"  kf_iou  p5/p50            : {np.percentile(k, 5):.2f} / {np.percentile(k, 50):.2f}")
        print(f"  fused_cost p95/max        : {np.percentile(fc, 95):.2f} / {fc.max():.2f}  "
              f"(ngưỡng ghép = {args.match_thresh})")
        near = np.mean(fc > args.match_thresh - 0.1)
        print(f"  % frame sát ngưỡng (>{args.match_thresh - 0.1:.1f})  : {near:.1%}")
        # Tương quan kf_iou với |yaw_cmd| -> quay càng nhanh IoU càng thấp?
        m = tracking & ~np.isnan(kf_iou)
        if m.sum() > 20:
            r = np.corrcoef(np.abs(yaw_cmd[m]), kf_iou[m])[0, 1]
            print(f"  corr(|yaw_cmd|, kf_iou)   : {r:+.2f}  (âm mạnh => quay nhanh làm hỏng phép ghép)")
            for lo, hi in ((0, 5), (5, 15), (15, 25), (25, 1e9)):
                mm = m & (np.abs(yaw_cmd) >= lo) & (np.abs(yaw_cmd) < hi)
                if mm.sum():
                    print(f"     |yaw| {lo:>3}-{hi if hi < 1e9 else 'max':>3} °/s : kf_iou median="
                          f"{np.median(kf_iou[mm]):.2f}  (n={mm.sum()})")
    else:
        print("  (không có dữ liệu kf_iou)")
    if tracking.any():
        print(f"  conf mục tiêu p5/p50      : {pct(conf[tracking], 5):.2f} / {pct(conf[tracking], 50):.2f}")
    print(f"  Frame có box bị lọc conf<ngưỡng: {np.mean(nlow > 0):.0%}")

    # ------------------------------------------------------------------ 3
    lost = [e for e in events if e["type"] == "TARGET_LOST"]
    switches = [e for e in events if e["type"] == "ID_SWITCH_SUSPECTED"]
    recov = [e for e in events if e["type"] == "RECOVERED_SAME_ID"]
    unlock = [e for e in events if e["type"] == "UNLOCKED_TIMEOUT"]
    print(f"\n[3] SỰ KIỆN: mất={len(lost)}  hồi phục cùng ID={len(recov)}  "
          f"ID mới (switch)={len(switches)}  unlock timeout={len(unlock)}")
    cause_counter = Counter()
    idx_by_frame = {int(r["frame_idx"]): i for i, r in enumerate(rows)}
    for e in lost:
        tags = [c.split(":")[0] for c in e["suspected_causes"]]
        cause_counter.update(tags)
        print(f"\n  --- LOST #{e['evt_id']} @frame {e['frame_idx']} ID #{e['locked_id']} "
              f"(ByteTrack: {e['bytetrack_state_now']})")
        for c in e["suspected_causes"]:
            print(f"      * {c}")
        sw = [s for s in switches if s.get("evt_id") == e["evt_id"]]
        for s in sw:
            print(f"      => ID mới #{s['new_id']} sau {s['after_s']}s, IoU với bbox cũ={s['iou_with_last_bbox']}")
        i = idx_by_frame.get(e["frame_idx"])
        if i is not None:
            print("      frame  err_x  yaw_cmd att_rate  ego_px  dx_px  kf_iou  cost  conf  evt")
            for r in rows[max(0, i - 10): i + 1]:
                print(f"      {r['frame_idx']:>5} {r['err_x']:>6} {r['yaw_cmd']:>7} {r['att_yawrate_dps']:>8} "
                      f"{r['ego_shift_px']:>7} {r['target_dx_px']:>6} {r['kf_iou']:>7} {r['fused_cost']:>5} "
                      f"{r['conf']:>5}  {r['event']}")

    # ------------------------------------------------------------------ 4
    print("\n[4] XẾP HẠNG NGUYÊN NHÂN NGHI NGỜ")
    if not cause_counter:
        print("  Không có sự kiện mất mục tiêu.")
    for c, n in cause_counter.most_common():
        print(f"  {n:>3}x  {c}")
    print("\nGợi ý đọc kết quả:")
    print("  - ASSOCIATION_FAIL + EGO_MOTION/KF_PRED_DRIFT chiếm đa số => do drone quay quá nhanh/giật")
    print("    (Kp lớn). Kalman của ByteTrack không biết camera quay => dự đoán sai => tạo ID mới.")
    print("  - YAW_OSCILLATION/SATURATION => giảm Kp, bỏ deadband cứng, thêm lọc/giới hạn gia tốc yaw.")
    print("  - LOW_CONF_FILTERED => hạ track_conf xuống 0.1 (để ByteTrack dùng vòng ghép low-score).")
    print("  - DETECTOR_MISS => vấn đề của YOLO (motion blur, người xoay ngang, quá nhỏ).")

    if args.plot:
        plot(args.log_dir, t, err, yaw_cmd, att_rate, ego, kf_iou, fcost, conf, found, rows, lost, args.match_thresh)


def plot(log_dir, t, err, yaw_cmd, att_rate, ego, kf_iou, fcost, conf, found, rows, lost, thr):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Chưa cài matplotlib.")
        return
    fi = {int(r["frame_idx"]): i for i, r in enumerate(rows)}
    lost_t = [t[fi[e["frame_idx"]]] for e in lost if e["frame_idx"] in fi]
    fig, ax = plt.subplots(5, 1, figsize=(14, 13), sharex=True)
    ax[0].plot(t, err, label="err_x (px)"); ax[0].axhspan(-15, 15, color="g", alpha=0.15, label="deadband")
    ax[0].set_ylabel("px"); ax[0].legend(loc="upper right")
    ax[1].plot(t, yaw_cmd, label="yaw_cmd (°/s)")
    ax[1].plot(t, att_rate, label="ATTITUDE yawspeed (°/s)", alpha=0.7)
    ax[1].set_ylabel("°/s"); ax[1].legend(loc="upper right")
    ax[2].plot(t, ego, label="ego_shift (px/frame)"); ax[2].set_ylabel("px"); ax[2].legend(loc="upper right")
    ax[3].plot(t, kf_iou, label="kf_iou"); ax[3].plot(t, fcost, label="fused_cost")
    ax[3].axhline(thr, color="r", ls="--", label="match_thresh"); ax[3].set_ylim(0, 1.05)
    ax[3].legend(loc="upper right")
    ax[4].plot(t, conf, label="conf"); ax[4].plot(t, found, label="target_found", alpha=0.6)
    ax[4].axhline(0.5, color="r", ls="--", label="conf_thresh"); ax[4].legend(loc="upper right")
    ax[4].set_xlabel("t (s)")
    for a in ax:
        for lt in lost_t:
            a.axvline(lt, color="r", alpha=0.5)
    fig.suptitle("Đường đỏ dọc = thời điểm mất ID")
    out = os.path.join(log_dir, "analysis.png")
    fig.tight_layout(); fig.savefig(out, dpi=110)
    print(f"\nĐã lưu biểu đồ: {out}")


if __name__ == "__main__":
    sys.exit(main())
