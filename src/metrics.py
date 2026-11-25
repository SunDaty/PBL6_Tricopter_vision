# src/metrics.py
"""
Module đo đạc & đánh giá hệ thống (bật bằng --debug):
FPS suy luận, độ trễ capture->bbox, chất lượng bám (mất track / nghi đổi ID),
đáp ứng bước & độ vọt lố yaw, độ trễ toàn hệ thống (ảnh -> lệnh -> đáp ứng drone).
Mọi mốc thời gian phía Pi dùng time.monotonic() (giây).
"""
from collections import deque
import csv
import math
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np


class RollingStats:
    """Lưu mẫu và tính mean / P50 / P95 / max. `window` dùng cho số liệu live."""

    def __init__(self, window: int = 300, max_keep: int = 200000):
        self._all: list[float] = []
        self._win: deque[float] = deque(maxlen=window)
        self._max_keep = max_keep

    def add(self, v: float):
        if len(self._all) < self._max_keep:
            self._all.append(float(v))
        self._win.append(float(v))

    @staticmethod
    def _summ(data) -> dict:
        if len(data) == 0:
            return {"n": 0, "mean": float("nan"), "p50": float("nan"),
                    "p95": float("nan"), "max": float("nan")}
        a = np.asarray(data, dtype=float)
        return {"n": len(a), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
                "p95": float(np.percentile(a, 95)), "max": float(a.max())}

    def live(self) -> dict:
        return self._summ(self._win)

    def total(self) -> dict:
        return self._summ(self._all)


class ClockSync:
    """Ước lượng offset đồng hồ Pixhawk -> Pi bằng MAVLink TIMESYNC (lọc theo RTT nhỏ nhất)."""

    def __init__(self, keep: int = 10):
        self._samples: deque[tuple[float, float]] = deque(maxlen=keep)  # (rtt_ns, offset_ns)

    def update(self, ts1_ns: int, tc1_ns: int, now_ns: int):
        rtt = now_ns - ts1_ns
        if rtt <= 0:
            return
        self._samples.append((float(rtt), tc1_ns - (ts1_ns + rtt / 2.0)))

    @property
    def ready(self) -> bool:
        return len(self._samples) > 0

    @property
    def rtt_ms(self) -> Optional[float]:
        return min(s[0] for s in self._samples) / 1e6 if self._samples else None

    def to_pi_time(self, fc_time_ns: float) -> Optional[float]:
        """Đổi thời gian Pixhawk (ns) sang đồng hồ monotonic của Pi (giây)."""
        if not self._samples:
            return None
        offset = min(self._samples, key=lambda s: s[0])[1]
        return (fc_time_ns - offset) / 1e9


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class IdQualityMonitor:
    """Đếm mất track ID khóa và nghi ngờ đổi ID (ID mới, IoU>=0.3 với bbox cuối, trong N khung)."""

    def __init__(self, window_frames: int = 10, iou_thresh: float = 0.3):
        self.window_frames = window_frames
        self.iou_thresh = iou_thresh
        self.lock_loss = 0
        self.suspected_switches = 0
        self.reacquire_count = 0
        self.known_ids: set[int] = set()
        self._cur_lock: Optional[int] = None
        self._last_bbox = None
        self._missing = 0
        self._known_at_loss: set[int] = set()
        self._switch_flagged = False

    def update(self, detections, locked_id: Optional[int], reacq_flag: bool = False) -> bool:
        """Trả True nếu khung này bị đánh dấu nghi đổi ID."""
        flag = False
        ids = {d.track_id for d in detections}
        if reacq_flag and locked_id is not None:
            self.reacquire_count += 1
            self._cur_lock = locked_id
            self._missing = 0
            self._switch_flagged = False
        elif locked_id != self._cur_lock:  # chọn mục tiêu mới / hủy khóa
            self._cur_lock = locked_id
            self._last_bbox, self._missing, self._switch_flagged = None, 0, False
        if locked_id is not None:
            target = next((d for d in detections if d.track_id == locked_id), None)
            if target is not None:
                self._last_bbox, self._missing, self._switch_flagged = target.bbox, 0, False
            elif self._last_bbox is not None:
                if self._missing == 0:
                    self.lock_loss += 1
                    self._known_at_loss = set(self.known_ids)
                self._missing += 1
                if self._missing <= self.window_frames and not self._switch_flagged:
                    for d in detections:
                        if d.track_id not in self._known_at_loss and \
                                _iou(d.bbox, self._last_bbox) >= self.iou_thresh:
                            self.suspected_switches += 1
                            self._switch_flagged = flag = True
                            break
        self.known_ids |= ids
        return flag


class YawStepAnalyzer:
    """Phát hiện 'bước' sai số ngang (|e| vượt ngưỡng) rồi đo rise / settling / overshoot."""

    def __init__(self, deadband_px: float, start_px: float = 60.0, timeout_s: float = 5.0,
                 settle_hold_s: float = 0.5, hfov_deg: float = 66.0, frame_width: int = 640):
        self.deadband = deadband_px
        self.start_px = start_px
        self.timeout_s = timeout_s
        self.settle_hold_s = settle_hold_s
        self.focal_px = (frame_width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        self.steps: list[dict] = []
        self._active: Optional[dict] = None
        self._prev_small = True

    def update(self, t: float, err_x: Optional[float], yaw_deg: Optional[float]):
        if err_x is None:
            self._active = None
            self._prev_small = True
            return
        a = self._active
        if a is None:
            if abs(err_x) >= self.start_px and self._prev_small:
                self._active = {"t0": t, "e0": err_x, "s": [(t, err_x, yaw_deg)], "in_db_since": None}
            self._prev_small = abs(err_x) < self.start_px
            return
        a["s"].append((t, err_x, yaw_deg))
        if abs(err_x) <= self.deadband:
            a["in_db_since"] = a["in_db_since"] if a["in_db_since"] is not None else t
        else:
            a["in_db_since"] = None
        settled = a["in_db_since"] is not None and t - a["in_db_since"] >= self.settle_hold_s
        if settled or t - a["t0"] >= self.timeout_s:
            self._finish(settled)
            self._active = None
            self._prev_small = abs(err_x) < self.start_px

    def _finish(self, settled: bool):
        a = self._active
        s, e0, t0 = a["s"], a["e0"], a["t0"]
        sign = 1.0 if e0 > 0 else -1.0
        thr = max(0.1 * abs(e0), self.deadband)
        rise = next((t - t0 for t, e, _ in s if abs(e) <= thr), float("nan"))
        settle = (a["in_db_since"] - t0) if settled else float("nan")
        over_px = max(0.0, max(-sign * e for _, e, _ in s))
        yaws = [y for _, _, y in s if y is not None]
        yaw_step = yaw_over = float("nan")
        if len(yaws) >= 2:
            u = np.degrees(np.unwrap(np.radians(yaws)))
            travel = u[-1] - u[0]
            d = 1.0 if travel >= 0 else -1.0
            yaw_step = float(abs(travel))
            yaw_over = float(max(0.0, max(d * (u - u[-1]))))
        self.steps.append({
            "t0": t0, "e0_px": e0, "e0_deg": math.degrees(math.atan(e0 / self.focal_px)),
            "rise_s": rise, "settle_s": settle, "overshoot_pct": over_px / abs(e0) * 100.0,
            "yaw_travel_deg": yaw_step, "yaw_overshoot_deg": yaw_over, "settled": settled,
        })


class CmdResponseTracker:
    """Đo cmd->đáp ứng drone (yawspeed) và e2e (capture->đáp ứng) khi lệnh yaw đổi đột ngột."""

    def __init__(self, jump_deg_s: float = 5.0, quiet_deg_s: float = 1.0, timeout_s: float = 2.0):
        self.jump, self.quiet, self.timeout = jump_deg_s, quiet_deg_s, timeout_s
        self._prev_cmd = 0.0
        self._pending: Optional[dict] = None
        self.cmd_to_resp = RollingStats()
        self.e2e = RollingStats()
        self.misses = 0

    def on_command(self, t_cmd: float, t_capture: float, yaw_cmd: float):
        if self._pending is not None and t_cmd - self._pending["t_cmd"] > self.timeout:
            self.misses += 1
            self._pending = None
        if self._pending is None and abs(self._prev_cmd) < self.quiet and abs(yaw_cmd) >= self.jump:
            self._pending = {"t_cmd": t_cmd, "t_cap": t_capture, "cmd": yaw_cmd}
        self._prev_cmd = yaw_cmd

    def on_attitude(self, t_pi: float, yawspeed_deg_s: float):
        p = self._pending
        if p is None or t_pi < p["t_cmd"]:
            return
        if yawspeed_deg_s * p["cmd"] > 0 and abs(yawspeed_deg_s) >= max(2.0, 0.2 * abs(p["cmd"])):
            self.cmd_to_resp.add((t_pi - p["t_cmd"]) * 1000.0)
            self.e2e.add((t_pi - p["t_cap"]) * 1000.0)
            self._pending = None


CSV_FIELDS = [
    "frame", "t_capture", "infer_ms", "cap2bbox_ms", "cap2cmd_ms", "locked_id", "n_det",
    "err_x", "yaw_cmd", "vx_cmd", "yaw_att_deg", "yawspeed_att", "id_switch_flag",
    "state", "err_deg_now", "psi_cap_deg", "reacq_flag",
]
STEP_FIELDS = ["t0", "e0_px", "e0_deg", "rise_s", "settle_s", "overshoot_pct",
               "yaw_travel_deg", "yaw_overshoot_deg", "settled"]


class DebugMetrics:
    def __init__(self, out_dir: str, deadband_px: float, hfov_deg: float = 66.0, frame_width: int = 640):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.clock_sync = ClockSync()
        self.infer_ms = RollingStats()
        self.cap2bbox = RollingStats()
        self.cap2cmd = RollingStats()
        self.id_mon = IdQualityMonitor()
        self.yaw = YawStepAnalyzer(deadband_px, hfov_deg=hfov_deg, frame_width=frame_width)
        self.resp = CmdResponseTracker()
        self._done_times: deque[float] = deque(maxlen=120)
        self._frames = 0
        self._t_start = time.monotonic()
        self._last_yaw: Optional[float] = None
        self._last_yawspeed: Optional[float] = None
        self._state_counts: dict[str, int] = {"IDLE": 0, "TRACK": 0, "COAST": 0, "SEARCH_HOLD": 0}
        self._csv_f = open(self.out_dir / "frames.csv", "w", newline="")
        self._csv = csv.DictWriter(self._csv_f, fieldnames=CSV_FIELDS)
        self._csv.writeheader()

    # --- Gọi từ luồng MAVLink: fc_time_ms = ATTITUDE.time_boot_ms (hoặc None) ---
    def on_attitude(self, t_arrival: float, yaw_deg: float, yawspeed_deg_s: float,
                    fc_time_ms: Optional[float] = None):
        t_pi = t_arrival
        if fc_time_ms is not None:
            conv = self.clock_sync.to_pi_time(fc_time_ms * 1e6)
            if conv is not None and abs(conv - t_arrival) < 1.0:  # chặn offset lỗi
                t_pi = conv
        with self._lock:
            self._last_yaw, self._last_yawspeed = yaw_deg, yawspeed_deg_s
            self.resp.on_attitude(t_pi, yawspeed_deg_s)

    def on_frame(self, frame_idx, t_capture, t_infer_start, t_infer_end, t_cmd_sent,
                 detections, locked_id, err_x, yaw_cmd, vx_cmd,
                 state: str = "", err_deg_now: Optional[float] = None,
                 psi_cap_deg: Optional[float] = None, reacq_flag: bool = False):
        with self._lock:
            self._frames += 1
            if state in self._state_counts:
                self._state_counts[state] += 1
            infer = (t_infer_end - t_infer_start) * 1000.0
            c2b = (t_infer_end - t_capture) * 1000.0
            c2c = (t_cmd_sent - t_capture) * 1000.0
            if self._frames > 1:  # bỏ khung đầu (warm-up YOLO) khỏi thống kê độ trễ
                self.infer_ms.add(infer)
                self.cap2bbox.add(c2b)
                self.cap2cmd.add(c2c)
            self._done_times.append(t_infer_end)
            flag = self.id_mon.update(detections, locked_id, reacq_flag=reacq_flag)
            self.yaw.update(t_infer_end, err_x, self._last_yaw)
            self.resp.on_command(t_cmd_sent, t_capture, yaw_cmd)
            self._csv.writerow({
                "frame": frame_idx, "t_capture": f"{t_capture:.4f}", "infer_ms": f"{infer:.2f}",
                "cap2bbox_ms": f"{c2b:.2f}", "cap2cmd_ms": f"{c2c:.2f}",
                "locked_id": "" if locked_id is None else locked_id, "n_det": len(detections),
                "err_x": "" if err_x is None else f"{err_x:.1f}", "yaw_cmd": f"{yaw_cmd:.2f}",
                "vx_cmd": f"{vx_cmd:.2f}",
                "yaw_att_deg": "" if self._last_yaw is None else f"{self._last_yaw:.2f}",
                "yawspeed_att": "" if self._last_yawspeed is None else f"{self._last_yawspeed:.2f}",
                "id_switch_flag": int(flag),
                "state": state,
                "err_deg_now": "" if err_deg_now is None else f"{err_deg_now:.2f}",
                "psi_cap_deg": "" if psi_cap_deg is None else f"{psi_cap_deg:.2f}",
                "reacq_flag": int(reacq_flag),
            })

    def fps_window(self) -> float:
        d = self._done_times
        if len(d) < 2 or d[-1] - d[0] <= 0:
            return 0.0
        return (len(d) - 1) / (d[-1] - d[0])

    def hud_lines(self) -> list[str]:
        with self._lock:
            inf, c2b = self.infer_ms.live(), self.cap2bbox.live()
            e2e = self.resp.e2e.live()
            last = self.yaw.steps[-1] if self.yaw.steps else None
            infer_fps = 1000.0 / inf["mean"] if inf["n"] and inf["mean"] > 0 else 0.0
            lines = [
                f"InferFPS: {infer_fps:.1f} | Loop: {self.fps_window():.1f}",
                f"Cap>bbox ms: {c2b['mean']:.0f} avg / {c2b['p95']:.0f} p95",
                f"ID loss: {self.id_mon.lock_loss} | ID sw?: {self.id_mon.suspected_switches}",
            ]
            if last is not None:
                lines.append(f"Yaw rise {last['rise_s']:.2f}s | OS {last['overshoot_pct']:.0f}%")
            else:
                lines.append("Yaw step: waiting")
            lines.append(f"E2E ms: {e2e['mean']:.0f} avg / {e2e['p95']:.0f} p95 (n={e2e['n']})")
            return lines

    def _state_summary_line(self) -> str:
        active = self._state_counts["TRACK"] + self._state_counts["COAST"] + self._state_counts["SEARCH_HOLD"]
        if active <= 0:
            return "Trạng thái bám: chưa khóa mục tiêu"
        tr = self._state_counts["TRACK"] / active * 100.0
        co = self._state_counts["COAST"] / active * 100.0
        sh = self._state_counts["SEARCH_HOLD"] / active * 100.0
        return f"Trạng thái khi khóa ({active} khung): TRACK {tr:.1f}% | COAST {co:.1f}% | HOLD {sh:.1f}%"

    def report(self) -> str:
        with self._lock:
            def row(name, s):
                return (f"{name:<28} n={s['n']:<6} mean={s['mean']:7.1f}  p50={s['p50']:7.1f}  "
                        f"p95={s['p95']:7.1f}  max={s['max']:7.1f}")
            inf = self.infer_ms.total()
            dur = max(1e-6, time.monotonic() - self._t_start)
            steps = self.yaw.steps
            L = ["=== BÁO CÁO DEBUG HỆ THỐNG ===",
                 f"Thời lượng: {dur:.1f}s | Số khung: {self._frames} | "
                 f"Pipeline FPS TB: {self._frames / dur:.1f} | "
                 f"Infer FPS (1/mean): {1000.0 / inf['mean'] if inf['n'] else 0:.1f}",
                 row("Infer (ms)", inf),
                 row("Capture->bbox (ms)", self.cap2bbox.total()),
                 row("Capture->cmd sent (ms)", self.cap2cmd.total()),
                 row("Cmd->drone response (ms)", self.resp.cmd_to_resp.total()),
                 row("E2E capture->response (ms)", self.resp.e2e.total()),
                 f"Phản hồi lệnh không bắt được (timeout): {self.resp.misses}",
                 f"Mất track ID khóa: {self.id_mon.lock_loss} | Nghi đổi ID: "
                 f"{self.id_mon.suspected_switches} | Tự bắt lại (RE-ACQ): "
                 f"{self.id_mon.reacquire_count} | ID duy nhất: {len(self.id_mon.known_ids)}",
                 self._state_summary_line(),
                 f"Đồng bộ đồng hồ: " + (f"OK, RTT min {self.clock_sync.rtt_ms:.2f} ms"
                                         if self.clock_sync.ready else "KHÔNG (dùng thời điểm nhận gói)"),
                 f"Số bước yaw đo được: {len(steps)}"]
            if steps:
                def m(k):
                    v = [s[k] for s in steps if not math.isnan(s[k])]
                    return float(np.mean(v)) if v else float("nan")
                L.append(f"  Rise TB {m('rise_s'):.2f}s | Settling TB {m('settle_s'):.2f}s | "
                         f"Overshoot TB {m('overshoot_pct'):.1f}% (max "
                         f"{max(s['overshoot_pct'] for s in steps):.1f}%) | "
                         f"Yaw overshoot TB {m('yaw_overshoot_deg'):.1f}°")
            return "\n".join(L)

    def close(self) -> str:
        text = self.report()
        with self._lock:
            self._csv_f.close()
            with open(self.out_dir / "steps.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=STEP_FIELDS)
                w.writeheader()
                for s in self.yaw.steps:
                    w.writerow({k: s[k] for k in STEP_FIELDS})
        (self.out_dir / "report.txt").write_text(text, encoding="utf-8")
        return text


if __name__ == "__main__":
    import tempfile
    from detector import DetectedPerson

    print("=== Kiểm thử độc lập: metrics.py ===")
    rs = RollingStats()
    for v in range(1, 101):
        rs.add(v)
    t = rs.total()
    assert abs(t["mean"] - 50.5) < 1e-6 and abs(t["p95"] - 95.05) < 1e-6
    print("[PASS] RollingStats mean/P95")

    cs = ClockSync()
    cs.update(ts1_ns=1_000_000_000, tc1_ns=6_000_000_000 + 1_000_000, now_ns=1_002_000_000)
    assert abs(cs.to_pi_time(6_000_000_000) - 1.0) < 1e-6 - 0 or abs(cs.to_pi_time(6e9) - 1.0) < 1e-3
    print("[PASS] ClockSync offset")

    mon = IdQualityMonitor()
    a = DetectedPerson(1, (100, 100, 200, 300), 0.9)
    mon.update([a], 1)
    mon.update([], 1)
    sw = mon.update([DetectedPerson(7, (105, 100, 205, 300), 0.9)], 1)
    assert mon.lock_loss == 1 and mon.suspected_switches == 1 and sw
    print("[PASS] IdQualityMonitor (mất track + nghi đổi ID)")

    # Đáp ứng bước tổng hợp: e0=200 -> vọt qua 0 tới -40 -> về 0
    ya = YawStepAnalyzer(deadband_px=15.0)
    ts = np.arange(0, 4.0, 0.05)
    for tt in ts:
        e = 200 * math.exp(-1.5 * tt) * math.cos(4 * tt)
        ya.update(float(tt), float(e), None)
    for tt in np.arange(4.0, 5.0, 0.05):
        ya.update(float(tt), 0.0, None)
    assert ya.steps, "Không phát hiện bước"
    s0 = ya.steps[0]
    assert s0["overshoot_pct"] > 5.0 and s0["settled"]
    print(f"[PASS] YawStepAnalyzer: rise={s0['rise_s']:.2f}s settle={s0['settle_s']:.2f}s "
          f"OS={s0['overshoot_pct']:.1f}%")

    cr = CmdResponseTracker()
    cr.on_command(t_cmd=10.0, t_capture=9.95, yaw_cmd=0.0)
    cr.on_command(t_cmd=10.1, t_capture=10.05, yaw_cmd=10.0)
    cr.on_attitude(10.15, 0.5)
    cr.on_attitude(10.2, 4.0)
    assert abs(cr.cmd_to_resp.total()["mean"] - 100.0) < 1e-6
    assert abs(cr.e2e.total()["mean"] - 150.0) < 1e-6
    print("[PASS] CmdResponseTracker (cmd->resp 100ms, e2e 150ms)")

    with tempfile.TemporaryDirectory() as d:
        dm = DebugMetrics(d, deadband_px=15.0)
        for i in range(5):
            tc = time.monotonic()
            dm.on_frame(i, tc, tc + 0.01, tc + 0.03, tc + 0.031, [a], 1, 20.0, 1.0, 0.0)
        assert "BÁO CÁO" in dm.close() and (Path(d) / "frames.csv").exists()
    print("[PASS] DebugMetrics CSV + report")
    print("=== Module metrics.py hoạt động tốt! ===")
