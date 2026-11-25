#!/usr/bin/env python3
"""
So sánh ByteTrack KHÔNG bù ego-motion (ngưỡng cũ 0.45) vs CÓ bù ego-motion (ngưỡng mới 0.20 + act 0.45)
trên kịch bản mô phỏng đúng hiện tượng log thực tế:
- 1 người đứng yên (hoặc đi ngang chậm)
- Drone lắc góc yaw ở 0.8 Hz, biên độ ±6° (~±52 px trên ảnh)
- 15% frame bị mất detection hoàn toàn + 20% frame bị mờ (conf = 0.28)
"""
import math
import sys
from pathlib import Path
import numpy as np
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ego_motion import BearingProjector


def run_scenario(use_ego_comp: bool, conf_thresh: float, act_thresh: float, match_thresh: float):
    rng = np.random.default_rng(42)
    proj = BearingProjector(frame_width=640, hfov_deg=66.0)
    tracker = sv.ByteTrack(
        track_activation_threshold=act_thresh,
        lost_track_buffer=30,
        minimum_matching_threshold=match_thresh,
        frame_rate=30,
    )

    n_frames = 150  # 15 giây @ 10 FPS
    seen_ids: set[int] = set()
    tracked_frames = 0
    id_changes = 0
    last_id = None

    for k in range(n_frames):
        t = k * 0.1
        # Drone lắc mạnh 0.8 Hz ±6° + quay chậm +15° tổng thể
        psi_deg = 6.0 * math.sin(2.0 * math.pi * 0.8 * t) + 1.0 * t
        psi_rad = math.radians(psi_deg)
        # Người đứng gần 8° thế giới, nhích rất chậm
        theta_person = math.radians(8.0 + 0.5 * math.sin(0.3 * t))
        xc = proj.bearing_to_px(theta_person, psi_rad) + float(rng.normal(0.0, 2.5))
        box = np.array([[xc - 35.0, 120.0, xc + 35.0, 340.0]], dtype=np.float32)

        r = rng.random()
        if r < 0.15:
            # Mất detection hoàn toàn (nhòe nặng)
            det = sv.Detections.empty()
        else:
            conf = 0.28 if r < 0.35 else 0.85
            if conf < conf_thresh:
                det = sv.Detections.empty()
            else:
                xyxy = proj.boxes_to_world(box, psi_rad) if use_ego_comp else box
                det = sv.Detections(
                    xyxy=xyxy,
                    confidence=np.array([conf], dtype=np.float32),
                    class_id=np.array([0], dtype=int),
                )

        tracked = tracker.update_with_detections(det)
        if len(tracked) > 0 and tracked.tracker_id is not None:
            tid = int(tracked.tracker_id[0])
            tracked_frames += 1
            seen_ids.add(tid)
            if last_id is not None and tid != last_id:
                id_changes += 1
            last_id = tid

    return len(seen_ids), id_changes, tracked_frames


if __name__ == "__main__":
    old_ids, old_sw, old_seen = run_scenario(
        use_ego_comp=False, conf_thresh=0.45, act_thresh=0.25, match_thresh=0.80
    )
    new_ids, new_sw, new_seen = run_scenario(
        use_ego_comp=True, conf_thresh=0.20, act_thresh=0.45, match_thresh=0.85
    )
    print(f"Cũ (conf>=0.45, không ego-comp): unique_IDs={old_ids}, id_switches={old_sw}, tracked={old_seen}/150")
    print(f"Mới (conf>=0.20, có ego-comp)  : unique_IDs={new_ids}, id_switches={new_sw}, tracked={new_seen}/150")

    assert new_ids == 1, f"Expected 1 unique ID with ego-motion comp, got {new_ids}"
    assert new_sw == 0, f"Expected 0 ID switches with ego-motion comp, got {new_sw}"
    assert new_seen > old_seen, "Expected higher tracked frame coverage with tier-2 + ego-motion comp"
    print("[PASS] ByteTrack + Ego-Motion giữ vững 1 ID duy nhất (0 lần đổi ID) trong dao động 0.8Hz ±6°!")
