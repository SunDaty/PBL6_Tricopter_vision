#!/usr/bin/env python3
"""Kiểm thử máy trạng thái TRACK/COAST/SEARCH_HOLD, RE-ACQUIRE, bù trễ, Hysteresis và Slew."""
import math
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from controller import HysteresisDeadband, SlewRateLimiter
from detector import DetectedPerson
from ego_motion import BearingProjector
from tracker import TargetTracker


def _make_frame_with_shirt(bbox, bgr_color):
    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr_color, -1)
    return frame


def test_hysteresis_and_slew():
    db = HysteresisDeadband(enter=15.0, exit=26.0)
    assert db.apply(30.0) == 15.0
    assert db.apply(10.0) == 0.0 and db.in_deadband
    assert db.apply(22.0) == 0.0 and db.in_deadband  # chưa vượt exit=26
    assert db.apply(-25.0) == 0.0 and db.in_deadband
    out = db.apply(30.0)
    assert abs(out - 15.0) < 1e-6 and not db.in_deadband

    slew = SlewRateLimiter(max_rate=60.0, initial_dt=0.1)
    v1 = slew.step(20.0, now=0.0)
    assert abs(v1 - 6.0) < 1e-6
    v2 = slew.step(20.0, now=0.1)
    assert abs(v2 - 12.0) < 1e-6
    v3 = slew.step(0.0, now=0.2)
    assert abs(v3 - 6.0) < 1e-6
    print("[PASS] HysteresisDeadband và SlewRateLimiter")


def test_latency_compensation():
    proj = BearingProjector(frame_width=640, hfov_deg=66.0)
    trk = TargetTracker(
        yaw_kp=0.08, yaw_kd=0.0, yaw_deadband_px=15.0, yaw_deadband_exit_px=26.0,
        max_yaw_rate=20.0, max_yaw_accel=200.0, latency_comp=True,
    )
    # Người đứng ở góc +10° thế giới. Lúc chụp ảnh drone đang ở yaw=0° -> trên ảnh người lệch phải ~87 px
    theta = math.radians(10.0)
    xc = proj.bearing_to_px(theta, 0.0)
    p = DetectedPerson(track_id=1, bbox=(int(xc - 40), 100, int(xc + 40), 340), conf=0.9)
    trk.select_by_point(int(xc), 220, [p], now=0.0, psi_capture=0.0)

    # Giả sử đến lúc tính lệnh (sau 150ms), drone ĐÃ xoay tới đúng +10°!
    # Bù trễ phải thấy sai số hiện tại = 0° -> lệnh yaw = 0 (không vọt lố!)
    cmd, active = trk.update([p], now=0.15, psi_capture=0.0, psi_now=math.radians(10.0))
    assert active is not None and trk.state == "TRACK"
    assert abs(cmd.yaw_rate) < 1e-6, f"Expected 0 yaw_rate after latency comp, got {cmd.yaw_rate}"
    assert abs(trk.last_error_deg_now) < 0.2
    print("[PASS] Bù độ trễ triệt tiêu lệnh thừa khi drone đã xoay tới mục tiêu")


def test_coast_and_hold_smooth():
    trk = TargetTracker(
        yaw_kp=0.08, yaw_kd=0.0, yaw_deadband_px=15.0, yaw_deadband_exit_px=26.0,
        max_yaw_rate=20.0, max_yaw_accel=60.0, coast_timeout=0.6, lost_timeout=2.0,
    )
    p = DetectedPerson(track_id=5, bbox=(460, 120, 540, 340), conf=0.9)  # cx=500 -> err=+180px
    trk.select_by_point(500, 220, [p], now=0.0, psi_capture=0.0)
    cmd0, _ = trk.update([p], now=0.1, psi_capture=0.0, psi_now=0.0)
    cmd1, _ = trk.update([p], now=0.2, psi_capture=0.0, psi_now=0.0)
    assert cmd1.yaw_rate > 8.0

    # Mất detection 0.2s -> COAST, lệnh KHÔNG rơi đột ngột về 0
    cmd_coast, active = trk.update([], now=0.4, psi_capture=0.0, psi_now=0.0)
    assert active is None and trk.state == "COAST"
    assert cmd_coast.yaw_rate > 2.0, f"COAST dropped to 0 too early: {cmd_coast.yaw_rate}"

    # Quá coast_timeout (0.6s) -> SEARCH_HOLD, giảm dần mượt về 0
    cmd_hold1, _ = trk.update([], now=0.9, psi_capture=0.0, psi_now=0.0)
    assert trk.state == "SEARCH_HOLD" and trk.locked_id == 5
    cmd_hold2, _ = trk.update([], now=1.2, psi_capture=0.0, psi_now=0.0)
    assert abs(cmd_hold2.yaw_rate) < 1e-6

    # Quá lost_timeout (2.0s kể từ t=0.2) -> IDLE
    cmd_idle, _ = trk.update([], now=2.5, psi_capture=0.0, psi_now=0.0)
    assert trk.state == "IDLE" and trk.locked_id is None and cmd_idle.yaw_rate == 0.0
    print("[PASS] Chuỗi TRACK -> COAST (mượt) -> SEARCH_HOLD -> IDLE")


def test_reacquire_and_gates():
    proj = BearingProjector(frame_width=640, hfov_deg=66.0)
    trk = TargetTracker(
        coast_timeout=0.6, lost_timeout=3.0,
        reacquire_enabled=True, reacquire_use_color_hist=True,
        single_person_fallback=True,
    )
    # 2 người ban đầu: #10 áo xanh dương ở góc +8°, #20 áo đỏ ở góc -15°
    xc1 = int(proj.bearing_to_px(math.radians(8.0), 0.0))
    xc2 = int(proj.bearing_to_px(math.radians(-15.0), 0.0))
    b1 = (xc1 - 40, 120, xc1 + 40, 340)
    b2 = (xc2 - 40, 120, xc2 + 40, 340)
    frame1 = np.full((480, 640, 3), 30, dtype=np.uint8)
    cv2.rectangle(frame1, (b1[0], b1[1]), (b1[2], b1[3]), (220, 60, 20), -1)  # xanh dương
    cv2.rectangle(frame1, (b2[0], b2[1]), (b2[2], b2[3]), (20, 30, 230), -1)  # đỏ

    p1 = DetectedPerson(track_id=10, bbox=b1, conf=0.9)
    p2 = DetectedPerson(track_id=20, bbox=b2, conf=0.9)
    trk.select_by_point(xc1, 220, [p1, p2], now=0.0, frame=frame1, psi_capture=0.0)
    trk.update([p1, p2], now=0.1, frame=frame1, psi_capture=0.0, psi_now=0.0)
    ref_h_before = trk.ref_height_px

    # Frame sau: drone quay sang +6° -> người #10 bị đổi ID thành #15, nằm ở góc +8° (ảnh dịch trái ~52px!)
    # Người #20 vẫn có mặt (thuộc coexisting_ids -> không được bắt nhầm!)
    psi_new = math.radians(6.0)
    xc1_new = int(proj.bearing_to_px(math.radians(8.0), psi_new))
    b1_new = (xc1_new - 40, 122, xc1_new + 40, 338)
    frame2 = np.full((480, 640, 3), 30, dtype=np.uint8)
    cv2.rectangle(frame2, (b1_new[0], b1_new[1]), (b1_new[2], b1_new[3]), (220, 60, 20), -1)
    cv2.rectangle(frame2, (b2[0], b2[1]), (b2[2], b2[3]), (20, 30, 230), -1)

    p1_reid = DetectedPerson(track_id=15, bbox=b1_new, conf=0.85)
    cmd, active = trk.update([p1_reid, p2], now=0.2, frame=frame2, psi_capture=psi_new, psi_now=psi_new)
    assert active is not None and trk.locked_id == 15
    assert trk.reacquire_count == 1 and trk.last_reacquired == (10, 15)
    assert trk.ref_height_px == ref_h_before, "RE-ACQ must preserve ref_height_px"

    # Nếu chỉ còn #20 (người khác trong coexisting_ids) -> TUYỆT ĐỐI không nhảy sang #20
    _, active_wrong = trk.update([p2], now=0.3, frame=frame2, psi_capture=psi_new, psi_now=psi_new)
    assert active_wrong is None and trk.locked_id == 15 and trk.state == "COAST"

    # Fallback 1 người duy nhất khi màu áo đổi mạnh (ngược sáng) nhưng đúng vùng vị trí & kích thước
    frame_backlit = _make_frame_with_shirt(b1_new, (20, 220, 220))  # màu vàng ngược sáng khác hẳn xanh
    p1_sole = DetectedPerson(track_id=99, bbox=b1_new, conf=0.7)
    _, active_fb = trk.update([p1_sole], now=0.4, frame=frame_backlit, psi_capture=psi_new, psi_now=psi_new)
    assert active_fb is not None and trk.locked_id == 99 and trk.reacquire_count == 2
    print("[PASS] RE-ACQUIRE (đúng vị trí/màu, giữ ref_h, từ chối người khác, fallback 1 người)")


if __name__ == "__main__":
    test_hysteresis_and_slew()
    test_latency_compensation()
    test_coast_and_hold_smooth()
    test_reacquire_and_gates()
    print("=== ALL tracker_states TESTS PASSED ===")
