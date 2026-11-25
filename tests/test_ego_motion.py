#!/usr/bin/env python3
"""Kiểm thử module ego_motion.py (YawHistory + BearingProjector)."""
import math
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ego_motion import BearingProjector, YawHistory, wrap_angle_rad


def test_yaw_history_unwrap_and_interp():
    yh = YawHistory(keep_s=2.0)
    # Qua biên +180 -> -180 độ
    yh.add(10.0, math.radians(178.0), math.radians(40.0))
    yh.add(10.1, math.radians(-178.0), math.radians(40.0))
    yh.add(10.2, math.radians(-174.0), math.radians(40.0))

    y_mid = yh.yaw_at(10.05)
    assert y_mid is not None
    assert abs(math.degrees(y_mid) - 180.0) < 1e-3, f"got {math.degrees(y_mid)}"

    y_after = yh.yaw_at(10.15)
    assert y_after is not None
    assert abs(math.degrees(y_after) - 184.0) < 1e-3, f"got {math.degrees(y_after)}"

    # Ngoại suy ngắn (<= 60ms) bằng yawspeed
    y_ext = yh.yaw_at(10.25, max_extrap_s=0.06)
    assert y_ext is not None
    assert abs(math.degrees(y_ext) - 188.0) < 1e-2, f"got {math.degrees(y_ext)}"

    # Quá hạn ngoại suy -> None
    assert yh.yaw_at(10.50, max_extrap_s=0.06) is None
    assert yh.yaw_at(9.50, max_extrap_s=0.06) is None

    # Qua biên ngược chiều (-179 -> +179)
    yh2 = YawHistory(keep_s=2.0)
    yh2.add(1.0, math.radians(-179.0), math.radians(-20.0))
    yh2.add(1.1, math.radians(179.0), math.radians(-20.0))
    y_back = yh2.yaw_at(1.05)
    assert y_back is not None and abs(math.degrees(y_back) - (-180.0)) < 1e-3
    print("[PASS] YawHistory unwrap ±180°, nội suy và giới hạn ngoại suy")


def test_bearing_projector_roundtrip():
    proj = BearingProjector(frame_width=640, hfov_deg=66.0)
    boxes = np.array(
        [
            [20.0, 50.0, 120.0, 350.0],
            [280.0, 100.0, 360.0, 420.0],
            [500.0, 80.0, 620.0, 460.0],
        ],
        dtype=np.float32,
    )
    for psi_deg in (-420.0, -175.0, -22.5, 0.0, 35.0, 210.0, 740.0):
        psi = math.radians(psi_deg)
        w = proj.boxes_to_world(boxes, psi)
        assert np.all(w[:, 2] > w[:, 0]), "BBox width must stay positive in world coords"
        back = proj.boxes_to_image(w, psi)
        err = float(np.max(np.abs(back - boxes)))
        assert err < 0.05, f"Roundtrip error {err} px at psi={psi_deg}°"

    # Kiểm tra bất biến góc thế giới: người cố định ở theta=+10°, drone quay từ 0° -> +6°
    theta_person = math.radians(10.0)
    x_psi0 = proj.bearing_to_px(theta_person, 0.0)
    x_psi6 = proj.bearing_to_px(theta_person, math.radians(6.0))
    assert x_psi6 < x_psi0, "Khi drone quay phải (+6°), người trên ảnh phải dịch sang trái"
    th_back = proj.px_to_bearing(x_psi6, math.radians(6.0))
    assert abs(wrap_angle_rad(th_back - theta_person)) < 1e-5
    print("[PASS] BearingProjector thuận-nghịch (< 0.05 px) và bất biến phương vị thế giới")


if __name__ == "__main__":
    test_yaw_history_unwrap_and_interp()
    test_bearing_projector_roundtrip()
    print("=== ALL ego_motion TESTS PASSED ===")
