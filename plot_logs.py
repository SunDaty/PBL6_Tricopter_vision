#!/usr/bin/env python3
"""
Xuất đồ thị đánh giá từ file CSV trong thư mục logs/ (mặc định chọn thư mục mới nhất).

Cách dùng:
    /home/dat/drone_tracking/drone_env/bin/python plot_logs.py
    /home/dat/drone_tracking/drone_env/bin/python plot_logs.py --log-dir logs/20261117_212706
"""
import argparse
import csv
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _to_float(val: Optional[str]) -> float:
    if val is None:
        return float("nan")
    s = val.strip()
    if not s:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def find_latest_log_dir(logs_root: Path) -> Path:
    candidates = [
        p for p in logs_root.iterdir()
        if p.is_dir() and (p / "frames.csv").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"Không tìm thấy thư mục log nào chứa frames.csv trong {logs_root}")
    return max(candidates, key=lambda p: p.name)


def load_frames_csv(csv_path: Path) -> dict[str, np.ndarray]:
    cols = [
        "frame", "t_capture", "infer_ms", "cap2bbox_ms", "cap2cmd_ms",
        "locked_id", "n_det", "err_x", "yaw_cmd", "vx_cmd",
        "yaw_att_deg", "yawspeed_att", "id_switch_flag",
        "err_deg_now", "psi_cap_deg", "reacq_flag",
    ]
    data: dict[str, list[float]] = {c: [] for c in cols}
    states: list[str] = []
    skipped = 0

    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        cleaned_lines = (line.replace("\x00", "") for line in f)
        reader = csv.DictReader(cleaned_lines)
        for row in reader:
            if not row or row.get("frame") is None or row.get("t_capture") is None:
                skipped += 1
                continue
            frame_v = _to_float(row.get("frame"))
            t_v = _to_float(row.get("t_capture"))
            if np.isnan(frame_v) or np.isnan(t_v):
                skipped += 1
                continue
            for c in cols:
                data[c].append(_to_float(row.get(c)))
            states.append((row.get("state") or "").strip())

    if skipped:
        print(f"[INFO] Đã bỏ qua {skipped} dòng không hợp lệ/bị cắt ở cuối file.")

    arrays: dict[str, np.ndarray] = {c: np.asarray(v, dtype=float) for c, v in data.items()}
    arrays["state"] = np.asarray(states, dtype=object)
    if len(arrays["t_capture"]) > 0:
        arrays["t_rel"] = arrays["t_capture"] - arrays["t_capture"][0]
    else:
        arrays["t_rel"] = np.array([], dtype=float)
    return arrays


def _shade_locked(ax, t: np.ndarray, locked: np.ndarray, states: Optional[np.ndarray] = None):
    is_locked = ~np.isnan(locked)
    if np.any(is_locked):
        ax.fill_between(
            t, 0, 1, where=is_locked,
            transform=ax.get_xaxis_transform(),
            color="tab:green", alpha=0.08, label="Target locked",
        )
    if states is not None and len(states) == len(t):
        is_coast = states == "COAST"
        if np.any(is_coast):
            ax.fill_between(
                t, 0, 1, where=is_coast,
                transform=ax.get_xaxis_transform(),
                color="tab:orange", alpha=0.18, label="COAST",
            )


def plot_overview(d: dict[str, np.ndarray], out_path: Path, title_prefix: str):
    t = d["t_rel"]
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    fig.suptitle(f"{title_prefix} — Tổng quan hệ thống", fontsize=13, fontweight="bold")

    # 1. Tracking error
    ax = axes[0]
    _shade_locked(ax, t, d["locked_id"])
    ax.plot(t, d["err_x"], color="tab:red", lw=1.1, label="err_x (px)")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_ylabel("err_x (px)")
    ax.set_title("Sai số bám ngang (err_x)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # 2. Yaw command & attitude
    ax = axes[1]
    _shade_locked(ax, t, d["locked_id"])
    l1 = ax.plot(t, d["yaw_cmd"], color="tab:blue", lw=1.1, label="yaw_cmd")
    l2 = ax.plot(t, d["yawspeed_att"], color="tab:orange", lw=1.0, alpha=0.85, label="yawspeed_att")
    ax.set_ylabel("Yaw rate")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    l3 = ax2.plot(t, d["yaw_att_deg"], color="tab:purple", lw=0.9, alpha=0.65, label="yaw_att_deg")
    ax2.set_ylabel("yaw_att (deg)", color="tab:purple")
    lines = l1 + l2 + l3
    ax.legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=8)
    ax.set_title("Điều khiển Yaw & Đáp ứng góc Yaw")

    # 3. Forward command & Detections
    ax = axes[2]
    l1 = ax.plot(t, d["vx_cmd"], color="tab:green", lw=1.1, label="vx_cmd")
    ax.set_ylabel("vx_cmd")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    l2 = ax2.step(t, d["n_det"], where="post", color="tab:brown", lw=0.9, alpha=0.7, label="n_det")
    ax2.set_ylabel("n_det", color="tab:brown")
    lines = l1 + l2
    ax.legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=8)
    ax.set_title("Lệnh tiến (vx_cmd) & Số lượng detection (n_det)")

    # 4. Latencies
    ax = axes[3]
    ax.plot(t, d["infer_ms"], lw=1.0, label="infer_ms")
    ax.plot(t, d["cap2bbox_ms"], lw=1.0, label="cap2bbox_ms")
    ax.plot(t, d["cap2cmd_ms"], lw=1.0, alpha=0.85, label="cap2cmd_ms")
    ax.set_ylabel("Latency (ms)")
    ax.set_xlabel("Thời gian từ lúc bắt đầu (s)")
    ax.set_title("Độ trễ xử lý")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_latency(d: dict[str, np.ndarray], out_path: Path, title_prefix: str):
    t = d["t_rel"]
    dt = np.diff(t)
    valid_dt = dt[(dt > 0) & (dt < 2.0)]
    fps = 1.0 / valid_dt if len(valid_dt) else np.array([])
    t_fps = t[1:][(dt > 0) & (dt < 2.0)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"{title_prefix} — Phân tích độ trễ & FPS", fontsize=13, fontweight="bold")

    # Top-left: latency time series
    ax = axes[0, 0]
    for col, label in [("infer_ms", "infer_ms"), ("cap2bbox_ms", "cap2bbox_ms"), ("cap2cmd_ms", "cap2cmd_ms")]:
        ax.plot(t, d[col], lw=0.9, label=label)
    ax.set_title("Độ trễ theo thời gian")
    ax.set_xlabel("Thời gian (s)")
    ax.set_ylabel("ms")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Top-right: latency histogram
    ax = axes[0, 1]
    for col, label in [("infer_ms", "infer_ms"), ("cap2bbox_ms", "cap2bbox_ms"), ("cap2cmd_ms", "cap2cmd_ms")]:
        vals = d[col][~np.isnan(d[col])]
        if len(vals):
            p50, p95 = np.percentile(vals, [50, 95])
            ax.hist(vals, bins=40, alpha=0.45, label=f"{label} (P50={p50:.1f}, P95={p95:.1f})")
    ax.set_title("Phân bố độ trễ")
    ax.set_xlabel("ms")
    ax.set_ylabel("Số frame")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Bottom-left: FPS time series
    ax = axes[1, 0]
    if len(fps):
        ax.plot(t_fps, fps, color="teal", lw=0.8, alpha=0.8, label=f"FPS (mean={fps.mean():.2f})")
        ax.axhline(fps.mean(), color="tab:red", ls="--", lw=1.0)
    ax.set_title("Tốc độ khung hình tức thời (FPS)")
    ax.set_xlabel("Thời gian (s)")
    ax.set_ylabel("FPS")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Bottom-right: Boxplot summary
    ax = axes[1, 1]
    box_data = []
    box_labels = []
    for col in ["infer_ms", "cap2bbox_ms", "cap2cmd_ms"]:
        vals = d[col][~np.isnan(d[col])]
        if len(vals):
            box_data.append(vals)
            box_labels.append(col)
    if box_data:
        ax.boxplot(box_data, labels=box_labels, showfliers=False)
    ax.set_title("Thống kê độ trễ (ẩn ngoại lệ)")
    ax.set_ylabel("ms")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_tracking_control(d: dict[str, np.ndarray], out_path: Path, title_prefix: str):
    t = d["t_rel"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(f"{title_prefix} — Chi tiết bám mục tiêu & Điều khiển", fontsize=13, fontweight="bold")

    # 1. Locked ID, ID switch & RE-ACQ
    ax = axes[0]
    ax.step(t, d["locked_id"], where="post", color="tab:blue", lw=1.2, label="locked_id")
    switches = np.where(d["id_switch_flag"] == 1)[0]
    if len(switches):
        ax.scatter(t[switches], d["locked_id"][switches], color="red", marker="x", s=50, zorder=5, label="id_switch_flag=1")
    reacqs = np.where(d["reacq_flag"] == 1)[0]
    if len(reacqs):
        ax.scatter(t[reacqs], d["locked_id"][reacqs], color="green", marker="^", s=55, zorder=6, label="RE-ACQ")
    ax.set_ylabel("Track ID")
    ax.set_title(f"Trạng thái khóa mục tiêu (nghi đổi ID: {len(switches)} | tự bắt lại RE-ACQ: {len(reacqs)})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # 2. err_x vs yaw_cmd
    ax = axes[1]
    _shade_locked(ax, t, d["locked_id"], d.get("state"))
    l1 = ax.plot(t, d["err_x"], color="tab:red", lw=1.1, label="err_x (px)")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_ylabel("err_x (px)", color="tab:red")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    l2 = ax2.plot(t, d["yaw_cmd"], color="tab:blue", lw=1.0, alpha=0.8, label="yaw_cmd")
    ax2.set_ylabel("yaw_cmd", color="tab:blue")
    lines = l1 + l2
    ax.legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=8)
    ax.set_title("Sai số ngang (err_x) & Lệnh Yaw (yaw_cmd)")

    # 3. vx_cmd & yawspeed_att
    ax = axes[2]
    _shade_locked(ax, t, d["locked_id"])
    ax.plot(t, d["vx_cmd"], color="tab:green", lw=1.1, label="vx_cmd")
    ax.plot(t, d["yawspeed_att"], color="tab:orange", lw=1.0, alpha=0.85, label="yawspeed_att")
    ax.set_ylabel("Giá trị")
    ax.set_xlabel("Thời gian từ lúc bắt đầu (s)")
    ax.set_title("Lệnh tiến (vx_cmd) & Tốc độ xoay thực tế (yawspeed_att)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Xuất đồ thị từ các file CSV trong thư mục logs.")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Đường dẫn tới thư mục log (mặc định: thư mục mới nhất trong logs/).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Thư mục lưu ảnh PNG (mặc định: <log-dir>/plots).",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    log_dir = args.log_dir if args.log_dir else find_latest_log_dir(base_dir / "logs")
    if not log_dir.is_absolute():
        log_dir = (Path.cwd() / log_dir).resolve()

    frames_csv = log_dir / "frames.csv"
    if not frames_csv.is_file():
        raise FileNotFoundError(f"Không tìm thấy {frames_csv}")

    out_dir = args.out_dir if args.out_dir else (log_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Đang đọc dữ liệu từ: {frames_csv}")
    d = load_frames_csv(frames_csv)
    n_rows = len(d["frame"])
    if n_rows == 0:
        raise RuntimeError("frames.csv không có dữ liệu hợp lệ để vẽ.")

    duration_s = float(d["t_rel"][-1])
    title_prefix = f"Log {log_dir.name}"

    outputs = [
        out_dir / "01_overview.png",
        out_dir / "02_latency_distribution.png",
        out_dir / "03_tracking_control.png",
    ]
    plot_overview(d, outputs[0], title_prefix)
    plot_latency(d, outputs[1], title_prefix)
    plot_tracking_control(d, outputs[2], title_prefix)

    print(f"Đã xử lý {n_rows} frames (thời lượng ~{duration_s:.1f}s).")
    for col in ["infer_ms", "cap2bbox_ms", "cap2cmd_ms"]:
        vals = d[col][~np.isnan(d[col])]
        if len(vals):
            print(
                f"  - {col}: mean={vals.mean():.1f}ms, "
                f"P50={np.percentile(vals, 50):.1f}ms, "
                f"P95={np.percentile(vals, 95):.1f}ms, max={vals.max():.1f}ms"
            )
    print("Các file đồ thị đã xuất:")
    for p in outputs:
        print(f"  -> {p}")


if __name__ == "__main__":
    main()
