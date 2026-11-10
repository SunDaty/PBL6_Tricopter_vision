# src/debug_logger.py
"""
[DEBUG] Bộ ghi log chẩn đoán hiện tượng MẤT ID mục tiêu.

Mỗi frame ghi 1 dòng CSV (frames.csv) + phát hiện sự kiện (events.jsonl) và lưu ảnh
các frame ngay trước/sau khi mất ID (events/evt_XXX/*.jpg).

Các đại lượng then chốt để tìm nguyên nhân:
  ego_shift_px   : Dịch chuyển ngang của ảnh do CHÍNH drone quay (đo từ ATTITUDE.yaw).
                   = -Δyaw(deg) * px_per_deg. Camera gắn cứng vào thân -> drone quay 1° thì
                   mọi vật trong ảnh trượt ~8px (HFOV 80°, 640px).
  target_dx_px   : Dịch chuyển tâm bbox mục tiêu giữa 2 frame xử lý liên tiếp.
  kf_iou         : IoU giữa bbox Kalman ByteTrack DỰ ĐOÁN (frame trước) và bbox thực tế.
                   Đây chính là đại lượng ByteTrack dùng để ghép ID.
  fused_cost     : 1 - kf_iou * conf  (fuse_score=True). ByteTrack CHỈ ghép nếu < match_thresh (0.8)
                   => cần kf_iou * conf > 0.2. Với conf=0.55 cần IoU > 0.36!
  yaw_sign_flips : Số lần đổi dấu lệnh yaw trong 2s gần nhất -> phát hiện dao động (Kp quá lớn).
"""
import csv
import json
import math
import os
import time
from collections import deque
from typing import Optional

import cv2


def iou_xyxy(a, b) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = max(1e-6, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    return inter / union


def _center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _wrap_deg(d: float) -> float:
    return (d + 180.0) % 360.0 - 180.0


class TrackingDebugLogger:
    CSV_FIELDS = [
        "t", "frame_idx", "rx_count", "dropped_since_last", "loop_dt_ms", "infer_ms",
        "state", "locked_id", "target_found", "cx", "cy", "bw", "bh", "conf", "edge_touch",
        "err_x", "p_term", "d_term", "yaw_raw", "yaw_cmd", "saturated", "in_deadband",
        "att_yaw_deg", "att_yawrate_dps", "att_roll_deg", "ego_shift_px",
        "target_dx_px", "target_dx_world_px", "iou_prev", "kf_iou", "fused_cost",
        "kf_state", "kf_vx", "n_dets", "n_filtered_lowconf", "n_tracked", "n_lost",
        "yaw_sign_flips_2s", "sat_ratio_2s", "event", "dets_json",
    ]

    def __init__(
        self,
        out_dir: Optional[str] = None,
        frame_width: int = 640,
        frame_height: int = 480,
        hfov_deg: float = 80.0,
        match_thresh: float = 0.8,
        ring_size: int = 20,
        post_event_frames: int = 10,
        save_frames: bool = True,
        print_events: bool = True,
    ):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.out_dir = out_dir or os.path.join(os.getcwd(), "debug_logs", stamp)
        os.makedirs(self.out_dir, exist_ok=True)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.px_per_deg = frame_width / hfov_deg
        self.match_thresh = match_thresh
        self.save_frames = save_frames
        self.print_events = print_events
        self.post_event_frames = post_event_frames

        self._csv_f = open(os.path.join(self.out_dir, "frames.csv"), "w", newline="")
        self._csv = csv.DictWriter(self._csv_f, fieldnames=self.CSV_FIELDS)
        self._csv.writeheader()
        self._evt_f = open(os.path.join(self.out_dir, "events.jsonl"), "w")

        self._ring = deque(maxlen=ring_size)        # (frame_idx, annotated_frame)
        self._hist = deque(maxlen=300)              # (t, yaw_cmd, saturated)
        self._pending_dumps: list[dict] = []        # sự kiện đang chờ lưu thêm frame sau

        self._prev = None                           # thông tin frame trước
        self._prev_snapshot: dict = {}
        self._seen_ids: set[int] = set()
        self._lost_ctx: Optional[dict] = None       # ngữ cảnh khi vừa mất mục tiêu
        self._evt_counter = 0
        self._prev_rx = 0
        self._prev_state = None

        print(f"[DEBUG] Ghi log chẩn đoán vào: {self.out_dir}")
        print(f"[DEBUG] px_per_deg = {self.px_per_deg:.2f} px/° "
              f"(1° yaw ~ {self.px_per_deg:.1f}px trượt ảnh)")

    # ------------------------------------------------------------------ #
    def log_frame(
        self,
        *,
        frame,
        frame_idx: int,
        rx_count: int,
        infer_ms: float,
        detections: list,
        filtered_dets: list,
        snapshot: dict,
        state: str,
        locked_id: Optional[int],
        target,
        center_x: float,
        yaw_pid,
        cmd,
        attitude: Optional[dict],
    ):
        t = time.monotonic()
        prev = self._prev
        event_tags: list[str] = []

        # ---------- Thời gian / rớt frame ----------
        loop_dt_ms = (t - prev["t"]) * 1000.0 if prev else 0.0
        dropped = max(0, rx_count - self._prev_rx - 1) if prev else 0
        self._prev_rx = rx_count

        # ---------- Chuyển động do drone tự quay ----------
        att_yaw = attitude.get("yaw_deg") if attitude else None
        att_rate = attitude.get("yawspeed_dps") if attitude else None
        att_roll = attitude.get("roll_deg") if attitude else None
        ego_shift = None
        if att_yaw is not None and prev and prev.get("att_yaw") is not None:
            ego_shift = -_wrap_deg(att_yaw - prev["att_yaw"]) * self.px_per_deg

        # ---------- Thông tin mục tiêu ----------
        row = {k: "" for k in self.CSV_FIELDS}
        bbox = target.bbox if target is not None else None
        cx = cy = bw = bh = conf = None
        target_dx = target_dx_world = iou_prev = kf_iou = fused_cost = None
        edge_touch = False
        if bbox is not None:
            cx, cy = _center(bbox)
            bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
            conf = target.conf
            edge_touch = bbox[0] <= 2 or bbox[2] >= self.frame_width - 2
            if prev and prev.get("target_id") == target.track_id and prev.get("bbox"):
                pcx, _ = _center(prev["bbox"])
                target_dx = cx - pcx
                iou_prev = iou_xyxy(prev["bbox"], bbox)
                if ego_shift is not None:
                    target_dx_world = target_dx - ego_shift
            pred = self._prev_snapshot.get(target.track_id, {}).get("pred_next")
            if pred is not None:
                kf_iou = iou_xyxy(pred, bbox)
                fused_cost = 1.0 - kf_iou * conf
                if fused_cost > self.match_thresh - 0.1:
                    event_tags.append("NEAR_MATCH_LIMIT")

        kf_info = snapshot.get(locked_id, {}) if locked_id is not None else {}

        # ---------- Thống kê dao động 2s ----------
        self._hist.append((t, cmd.yaw_rate, bool(yaw_pid.last_saturated)))
        recent = [h for h in self._hist if t - h[0] <= 2.0]
        signs = [math.copysign(1, h[1]) for h in recent if abs(h[1]) > 1e-6]
        flips = sum(1 for a, b in zip(signs, signs[1:]) if a != b)
        sat_ratio = sum(1 for h in recent if h[2]) / max(1, len(recent))

        # ---------- Phát hiện sự kiện ----------
        target_found = target is not None
        was_found = bool(prev and prev.get("target_found"))
        if state == "LOCKED" and locked_id is not None and locked_id < 0 and \
                (not prev or prev.get("target_id") != locked_id):
            event_tags.append("LOCKED_ON_FAKE_ID")
            self._emit({"type": "LOCKED_ON_FAKE_ID", "frame_idx": frame_idx, "locked_id": locked_id,
                        "hint": "Selector khoá vào ID âm (fallback của detector.py khi ByteTrack "
                                "không trả track nào). ID này sẽ biến mất ở frame sau -> mất mục tiêu."})
        if state == "LOCKED" and was_found and not target_found:
            event_tags.append("TARGET_LOST")
            self._on_target_lost(t, frame_idx, prev, detections, filtered_dets,
                                 snapshot, locked_id, flips, sat_ratio)
        if self._lost_ctx is not None:
            if target_found and target.track_id == self._lost_ctx["locked_id"]:
                event_tags.append("RECOVERED_SAME_ID")
                self._emit({"type": "RECOVERED_SAME_ID", "frame_idx": frame_idx,
                            "lost_frames": frame_idx - self._lost_ctx["frame_idx"],
                            "lost_s": round(t - self._lost_ctx["t"], 3)})
                self._lost_ctx = None
            else:
                self._check_id_switch(t, frame_idx, detections, event_tags)
        if self._prev_state == "LOCKED" and state == "WAITING_USER_SELECT":
            event_tags.append("UNLOCKED_TIMEOUT")
            self._emit({"type": "UNLOCKED_TIMEOUT", "frame_idx": frame_idx,
                        "hint": "Mất mục tiêu > lost_timeout -> drone hover chờ người dùng"})
            self._lost_ctx = None
        self._prev_state = state
        for d in detections:
            self._seen_ids.add(d.track_id)

        # ---------- Ghi CSV ----------
        row.update({
            "t": f"{t:.4f}", "frame_idx": frame_idx, "rx_count": rx_count,
            "dropped_since_last": dropped, "loop_dt_ms": f"{loop_dt_ms:.1f}",
            "infer_ms": f"{infer_ms:.1f}", "state": state,
            "locked_id": "" if locked_id is None else locked_id,
            "target_found": int(target_found),
            "cx": _f(cx), "cy": _f(cy), "bw": _f(bw), "bh": _f(bh), "conf": _f(conf, 3),
            "edge_touch": int(edge_touch),
            "err_x": _f(cx - center_x if cx is not None else None),
            "p_term": _f(yaw_pid.last_p, 2), "d_term": _f(yaw_pid.last_d, 2),
            "yaw_raw": _f(yaw_pid.last_raw, 2), "yaw_cmd": _f(cmd.yaw_rate, 2),
            "saturated": int(bool(yaw_pid.last_saturated)),
            "in_deadband": int(bool(yaw_pid.last_in_deadband)),
            "att_yaw_deg": _f(att_yaw, 2), "att_yawrate_dps": _f(att_rate, 2),
            "att_roll_deg": _f(att_roll, 2), "ego_shift_px": _f(ego_shift),
            "target_dx_px": _f(target_dx), "target_dx_world_px": _f(target_dx_world),
            "iou_prev": _f(iou_prev, 3), "kf_iou": _f(kf_iou, 3),
            "fused_cost": _f(fused_cost, 3),
            "kf_state": kf_info.get("state", ""), "kf_vx": _f(kf_info.get("vx"), 2),
            "n_dets": len(detections), "n_filtered_lowconf": len(filtered_dets),
            "n_tracked": sum(1 for s in snapshot.values() if s["state"] == "tracked"),
            "n_lost": sum(1 for s in snapshot.values() if s["state"] == "lost"),
            "yaw_sign_flips_2s": flips, "sat_ratio_2s": f"{sat_ratio:.2f}",
            "event": "|".join(event_tags),
            "dets_json": json.dumps([[d.track_id, round(d.conf, 2), list(d.bbox)] for d in detections]),
        })
        self._csv.writerow(row)
        if frame_idx % 30 == 0:
            self._csv_f.flush()
            self._evt_f.flush()

        # ---------- Lưu frame (ring buffer) ----------
        if self.save_frames:
            ann = self._annotate(frame, detections, filtered_dets, locked_id, target,
                                 self._prev_snapshot, frame_idx, event_tags, ego_shift, kf_iou)
            self._ring.append((frame_idx, ann))
            for dump in list(self._pending_dumps):
                cv2.imwrite(os.path.join(dump["dir"], f"f{frame_idx:06d}_post.jpg"), ann)
                dump["left"] -= 1
                if dump["left"] <= 0:
                    self._pending_dumps.remove(dump)

        self._prev = {
            "t": t, "target_found": target_found, "bbox": bbox,
            "target_id": target.track_id if target is not None else None,
            "att_yaw": att_yaw, "cmd": cmd.yaw_rate, "conf": conf,
            "ego_shift": ego_shift, "err_x": (cx - center_x) if cx is not None else None,
            "kf_iou": kf_iou, "loop_dt_ms": loop_dt_ms, "dropped": dropped,
            "edge_touch": edge_touch, "att_rate": att_rate,
        }
        self._prev_snapshot = snapshot

    # ------------------------------------------------------------------ #
    def _on_target_lost(self, t, frame_idx, prev, detections, filtered_dets,
                        snapshot, locked_id, flips, sat_ratio):
        """Phân loại nguyên nhân ngay tại frame mất mục tiêu."""
        last_bbox = prev["bbox"]
        pred = self._prev_snapshot.get(locked_id, {}).get("pred_next")
        kf_now = snapshot.get(locked_id)

        cands = []
        for d in detections:
            cands.append({
                "id": d.track_id, "conf": round(d.conf, 3), "bbox": list(d.bbox),
                "iou_last": round(iou_xyxy(last_bbox, d.bbox), 3),
                "iou_kf_pred": round(iou_xyxy(pred, d.bbox), 3) if pred else None,
                "dx_from_last": round(_center(d.bbox)[0] - _center(last_bbox)[0], 1),
                "new_id": d.track_id not in self._seen_ids,
            })
        low = [{"conf": round(c, 3), "bbox": list(b),
                "iou_last": round(iou_xyxy(last_bbox, b), 3)} for b, c in filtered_dets]

        near_det = [c for c in cands if c["iou_last"] > 0.1 or abs(c["dx_from_last"]) < 80]
        near_low = [c for c in low if c["iou_last"] > 0.1]

        causes = []
        if near_det:
            causes.append("ASSOCIATION_FAIL: có detection gần vị trí cũ nhưng mang ID khác "
                          "-> ByteTrack không ghép được (IoU với Kalman-pred quá thấp)")
        elif near_low:
            causes.append("LOW_CONF_FILTERED: YOLO vẫn thấy người (conf<ngưỡng) nhưng bị lọc "
                          "TRƯỚC khi vào ByteTrack (conf=0.5 vô hiệu hoá vòng ghép low-score)")
        else:
            causes.append("DETECTOR_MISS: YOLO không phát hiện được người ở frame này")
        if prev.get("ego_shift") is not None and abs(prev["ego_shift"]) > 0.3 * max(1.0, last_bbox[2] - last_bbox[0]):
            causes.append(f"EGO_MOTION: ảnh trượt {prev['ego_shift']:+.1f}px/frame do drone quay "
                          f"(> 30% bề rộng bbox {last_bbox[2]-last_bbox[0]}px)")
        if flips >= 3:
            causes.append(f"YAW_OSCILLATION: lệnh yaw đổi dấu {flips} lần/2s -> Kp quá lớn/trễ lớn")
        if sat_ratio > 0.5:
            causes.append(f"YAW_SATURATION: {sat_ratio:.0%} thời gian lệnh yaw bão hoà ±max_yaw_rate "
                          f"(Kp lớn -> điều khiển dạng bang-bang)")
        if prev.get("kf_iou") is not None and prev["kf_iou"] < 0.4:
            causes.append(f"KF_PRED_DRIFT: IoU Kalman-pred vs thực tế ở frame trước chỉ {prev['kf_iou']:.2f}")
        if prev.get("loop_dt_ms", 0) > 120 or prev.get("dropped", 0) > 1:
            causes.append(f"LOW_FPS/FRAME_DROP: dt={prev['loop_dt_ms']:.0f}ms, rớt {prev['dropped']} frame")
        if prev.get("edge_touch"):
            causes.append("FRAME_EDGE: bbox chạm mép ảnh (vọt lố đẩy người ra khỏi khung)")
        if prev.get("conf") is not None and prev["conf"] < 0.6:
            causes.append(f"CONF_DROPPING: conf frame trước chỉ {prev['conf']:.2f} (người xoay ngang?)")

        self._evt_counter += 1
        evt = {
            "type": "TARGET_LOST", "evt_id": self._evt_counter, "frame_idx": frame_idx,
            "locked_id": locked_id, "last_bbox": list(last_bbox),
            "last_conf": prev.get("conf"), "last_err_x": prev.get("err_x"),
            "last_yaw_cmd": prev.get("cmd"), "last_att_yawrate": prev.get("att_rate"),
            "last_ego_shift_px": prev.get("ego_shift"), "last_kf_iou": prev.get("kf_iou"),
            "kf_pred_bbox": [round(v, 1) for v in pred] if pred else None,
            "bytetrack_state_now": kf_now["state"] if kf_now else "removed/none",
            "yaw_sign_flips_2s": flips, "sat_ratio_2s": round(sat_ratio, 2),
            "candidates": cands, "filtered_lowconf": low, "suspected_causes": causes,
        }
        self._emit(evt)
        self._lost_ctx = {"t": t, "frame_idx": frame_idx, "locked_id": locked_id,
                          "last_bbox": last_bbox, "evt_id": self._evt_counter}

        if self.save_frames:
            d = os.path.join(self.out_dir, "events", f"evt_{self._evt_counter:03d}_id{locked_id}")
            os.makedirs(d, exist_ok=True)
            for fi, img in self._ring:
                cv2.imwrite(os.path.join(d, f"f{fi:06d}_pre.jpg"), img)
            self._pending_dumps.append({"dir": d, "left": self.post_event_frames})

    def _check_id_switch(self, t, frame_idx, detections, event_tags):
        ctx = self._lost_ctx
        if t - ctx["t"] > 3.0:
            self._lost_ctx = None
            return
        lb = ctx["last_bbox"]
        lw = max(1.0, lb[2] - lb[0])
        for d in detections:
            if d.track_id in self._seen_ids or d.track_id == ctx["locked_id"]:
                continue
            ov = iou_xyxy(lb, d.bbox)
            dx = _center(d.bbox)[0] - _center(lb)[0]
            # Chỉ coi là "cùng người đổi ID" nếu ID mới xuất hiện gần vị trí cũ
            if ov < 0.05 and abs(dx) > max(80.0, 2.0 * lw):
                continue
            neg = d.track_id < 0
            event_tags.append(f"NEW_ID_{d.track_id}")
            self._emit({
                "type": "ID_SWITCH_SUSPECTED", "frame_idx": frame_idx,
                "evt_id": ctx["evt_id"], "old_id": ctx["locked_id"], "new_id": d.track_id,
                "after_s": round(t - ctx["t"], 3), "iou_with_last_bbox": round(ov, 3),
                "dx_from_last_px": round(dx, 1), "conf": round(d.conf, 3),
                "hint": ("ID ÂM = ID giả của detector.py (ByteTrack chưa xác nhận track) -> "
                         "nếu selector khoá vào ID này sẽ mất ngay frame sau!") if neg else
                        ("ByteTrack đã tạo ID MỚI cho (nhiều khả năng) cùng một người; "
                         "selector đang chờ ID cũ nên drone hover rồi unlock."),
            })

    def _emit(self, evt: dict):
        self._evt_f.write(json.dumps(evt, ensure_ascii=False) + "\n")
        self._evt_f.flush()
        if not self.print_events:
            return
        typ = evt["type"]
        if typ == "TARGET_LOST":
            print(f"\n\033[91m[DEBUG][LOST #{evt['evt_id']}] frame {evt['frame_idx']} ID #{evt['locked_id']} "
                  f"| yaw_cmd={_f(evt['last_yaw_cmd'])} att_rate={_f(evt['last_att_yawrate'])} "
                  f"ego={_f(evt['last_ego_shift_px'])}px kf_iou={_f(evt['last_kf_iou'], 2)} "
                  f"conf={_f(evt['last_conf'], 2)} flips2s={evt['yaw_sign_flips_2s']} "
                  f"sat={evt['sat_ratio_2s']:.0%} BT={evt['bytetrack_state_now']}\033[0m")
            for c in evt["suspected_causes"]:
                print(f"   -> {c}")
        elif typ == "ID_SWITCH_SUSPECTED":
            print(f"\033[93m[DEBUG][ID SWITCH?] #{evt['old_id']} -> #{evt['new_id']} sau {evt['after_s']}s "
                  f"(IoU vs bbox cũ={evt['iou_with_last_bbox']}, dx={evt['dx_from_last_px']}px)\033[0m")
        else:
            print(f"\033[96m[DEBUG][{typ}] {json.dumps({k: v for k, v in evt.items() if k != 'type'}, ensure_ascii=False)}\033[0m")

    # ------------------------------------------------------------------ #
    def _annotate(self, frame, detections, filtered_dets, locked_id, target,
                  prev_snapshot, frame_idx, tags, ego_shift, kf_iou):
        img = frame.copy()
        for b, c in filtered_dets:  # box bị lọc vì conf thấp: xám
            cv2.rectangle(img, b[:2], b[2:], (128, 128, 128), 1)
            cv2.putText(img, f"low {c:.2f}", (b[0], b[3] + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
        for d in detections:
            col = (0, 255, 0) if d.track_id == locked_id else (0, 200, 255)
            cv2.rectangle(img, d.bbox[:2], d.bbox[2:], col, 2)
            cv2.putText(img, f"#{d.track_id} {d.conf:.2f}", (d.bbox[0], d.bbox[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        if locked_id is not None and locked_id in prev_snapshot:  # Kalman dự đoán: tím
            p = [int(v) for v in prev_snapshot[locked_id]["pred_next"]]
            cv2.rectangle(img, (p[0], p[1]), (p[2], p[3]), (255, 0, 255), 1)
            cv2.putText(img, "KF pred", (p[0], p[3] + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        txt = f"f{frame_idx} lock={locked_id} ego={_f(ego_shift)}px kf_iou={_f(kf_iou, 2)} {' '.join(tags)}"
        cv2.putText(img, txt, (5, self.frame_height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        return img

    def close(self):
        try:
            self._csv_f.close()
            self._evt_f.close()
        except Exception:
            pass
        print(f"[DEBUG] Đã lưu log: {self.out_dir}\n"
              f"        Phân tích: python3 debug/analyze_log.py {self.out_dir}")


def _f(v, nd: int = 1) -> str:
    return "" if v is None else f"{v:.{nd}f}"
