# src/selector.py
from enum import Enum
import time
from typing import Optional

from detector import DetectedPerson


class TrackingState(str, Enum):
    """Trạng thái của bộ lựa chọn và khóa mục tiêu."""
    AUTO_SEARCH = "AUTO_SEARCH"                  # Tự động tìm và khóa mục tiêu tối ưu nhất
    LOCKED = "LOCKED"                            # Đang khóa vào một track_id cụ thể
    WAITING_USER_SELECT = "WAITING_USER_SELECT"  # Hover đứng yên chờ người dùng chọn mục tiêu


class TargetSelector:
    """
    Module độc lập quản lý việc lựa chọn và khóa mục tiêu (Target Selection & Locking)
    khi có nhiều người trong khung hình:
    - Chấm điểm tự động ban đầu dựa trên khoảng cách tới tâm màn hình + diện tích khung bao.
    - Duy trì ID mục tiêu đang khóa (ByteTrack ID persistence).
    - Hỗ trợ chọn mục tiêu qua Click chuột hoặc chuyển đổi bằng bàn phím (Tab/N).
    - Tự động chuyển sang WAITING_USER_SELECT (Hover chờ người dùng) nếu mất dấu > lost_timeout (1-3s).
    """
    def __init__(
        self,
        frame_width: int = 640,
        frame_height: int = 480,
        lost_timeout: float = 2.0,
        center_weight: float = 0.6,
        area_weight: float = 0.4,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.center_x = frame_width / 2.0
        self.lost_timeout = lost_timeout
        self.center_weight = center_weight
        self.area_weight = area_weight

        self.state: TrackingState = TrackingState.AUTO_SEARCH
        self.locked_id: Optional[int] = None
        self.last_seen_time: Optional[float] = None
        self.last_known_bbox: Optional[tuple[int, int, int, int]] = None

    def reset(self):
        """Đặt lại về trạng thái tự động tìm kiếm ban đầu."""
        self.state = TrackingState.AUTO_SEARCH
        self.locked_id = None
        self.last_seen_time = None
        self.last_known_bbox = None

    def unlock(self):
        """Hủy khóa mục tiêu hiện tại và chuyển sang trạng thái Hover chờ người dùng chọn."""
        self.locked_id = None
        self.last_seen_time = None
        self.last_known_bbox = None
        self.state = TrackingState.WAITING_USER_SELECT

    def trigger_auto_select(self):
        """Kích hoạt chọn tự động mục tiêu tối ưu nhất (Gần tâm + Diện tích lớn)."""
        self.locked_id = None
        self.last_seen_time = None
        self.last_known_bbox = None
        self.state = TrackingState.AUTO_SEARCH

    def lock_id(self, track_id: int, now: Optional[float] = None) -> bool:
        """
        Khóa trực tiếp vào một track_id cụ thể.
        Trả về True nếu đổi sang một ID mới (để bộ điều khiển reset PID).
        """
        current_time = now if now is not None else time.monotonic()
        changed = self.locked_id != track_id
        self.locked_id = track_id
        self.last_seen_time = current_time
        self.state = TrackingState.LOCKED
        return changed

    def select_by_point(
        self,
        x: int,
        y: int,
        detections: list[DetectedPerson],
        now: Optional[float] = None,
    ) -> Optional[int]:
        """
        Chọn mục tiêu tại tọa độ click chuột (x, y).
        Nếu nhiều khung bao chồng lấn tại điểm click, ưu tiên người có tâm gần điểm click nhất.
        """
        candidates = [d for d in detections if d.contains_point(x, y)]
        if not candidates:
            return None

        chosen = min(
            candidates,
            key=lambda d: (d.center[0] - x) ** 2 + (d.center[1] - y) ** 2,
        )
        self.lock_id(chosen.track_id, now=now)
        self.last_known_bbox = chosen.bbox
        return chosen.track_id

    def cycle_target(
        self,
        detections: list[DetectedPerson],
        now: Optional[float] = None,
    ) -> Optional[int]:
        """Chuyển đổi luân phiên sang người tiếp theo trong khung hình (phím Tab / N)."""
        if not detections:
            return None

        sorted_dets = sorted(detections, key=lambda d: d.track_id)
        if self.locked_id is None:
            chosen = self.pick_best_candidate(sorted_dets)
        else:
            ids = [d.track_id for d in sorted_dets]
            if self.locked_id in ids:
                next_idx = (ids.index(self.locked_id) + 1) % len(sorted_dets)
                chosen = sorted_dets[next_idx]
            else:
                chosen = sorted_dets[0]

        self.lock_id(chosen.track_id, now=now)
        self.last_known_bbox = chosen.bbox
        return chosen.track_id

    def score_candidate(self, person: DetectedPerson, max_area: float) -> float:
        """
        Chấm điểm mục tiêu kết hợp:
        - Độ gần tâm ngang của màn hình (center_score in [0, 1])
        - Tỉ lệ diện tích khung bao so với người lớn nhất (area_score in [0, 1])
        """
        cx, _ = person.center
        half_w = max(1.0, self.frame_width / 2.0)
        norm_dist_center = min(1.0, abs(cx - self.center_x) / half_w)
        center_score = 1.0 - norm_dist_center

        area_score = (person.area / max_area) if max_area > 0 else 0.0
        return self.center_weight * center_score + self.area_weight * area_score

    def pick_best_candidate(self, detections: list[DetectedPerson]) -> DetectedPerson:
        """Chọn người có điểm kết hợp (Gần tâm + Gần drone) cao nhất."""
        max_area = float(max((d.area for d in detections), default=1))
        return max(detections, key=lambda d: self.score_candidate(d, max_area))

    def get_occlusion_elapsed(self, now: Optional[float] = None) -> float:
        """Thời gian (giây) kể từ lần cuối nhìn thấy mục tiêu đang khóa."""
        if self.state != TrackingState.LOCKED or self.last_seen_time is None:
            return 0.0
        current_time = now if now is not None else time.monotonic()
        return max(0.0, current_time - self.last_seen_time)

    def update(
        self,
        detections: list[DetectedPerson],
        now: Optional[float] = None,
    ) -> tuple[Optional[DetectedPerson], bool]:
        """
        Cập nhật danh sách phát hiện ở frame hiện tại và trả về:
        (DetectedPerson_được_chọn_hoặc_None, cờ_cần_reset_PID).
        """
        current_time = now if now is not None else time.monotonic()

        # 1. Trạng thái Hover chờ người dùng chọn mục tiêu:
        if self.state == TrackingState.WAITING_USER_SELECT:
            return None, True

        # 2. Trạng thái Tự động tìm kiếm ban đầu (hoặc khi người dùng bấm 'A'):
        if self.state == TrackingState.AUTO_SEARCH:
            if not detections:
                return None, True

            best = self.pick_best_candidate(detections)
            changed = self.lock_id(best.track_id, now=current_time)
            self.last_known_bbox = best.bbox
            return best, changed

        # 3. Trạng thái Đang khóa mục tiêu (LOCKED):
        target = next((d for d in detections if d.track_id == self.locked_id), None)
        if target is not None:
            self.last_seen_time = current_time
            self.last_known_bbox = target.bbox
            return target, False

        # Không thấy mục tiêu đang khóa trong frame này -> kiểm tra thời gian bị khuất
        elapsed = (
            current_time - self.last_seen_time
            if self.last_seen_time is not None
            else self.lost_timeout + 1.0
        )

        if elapsed <= self.lost_timeout:
            # Khuất ngắn hạn (<= lost_timeout): Giữ LOCKED nhưng trả về None để drone Hover chờ
            return None, True

        # Khuất quá thời gian quy định (> 1-3s):
        # Hủy khóa và chuyển sang WAITING_USER_SELECT (Hover chờ người dùng chọn lại)
        self.unlock()
        return None, True


if __name__ == "__main__":
    print("=== Kiểm thử độc lập Module 2: selector.py (TargetSelector) ===")
    selector = TargetSelector(lost_timeout=2.0)
    p1 = DetectedPerson(track_id=1, bbox=(290, 100, 360, 360), conf=0.9)
    p2 = DetectedPerson(track_id=2, bbox=(30, 80, 120, 350), conf=0.88)

    target, _ = selector.update([p1, p2], now=0.0)
    assert target is not None and target.track_id == 1
    print(f"[PASS] Tự động chọn mục tiêu gần tâm: ID #{target.track_id}")

    # Khuất 1.0s (<= 2.0s)
    target, _ = selector.update([p2], now=1.0)
    assert target is None and selector.state == TrackingState.LOCKED
    print(f"[PASS] Khuất ngắn hạn 1.0s -> Giữ khóa ID #{selector.locked_id}, trả về None để Hover")

    # Khuất 2.5s (> 2.0s)
    target, _ = selector.update([p2], now=2.5)
    assert target is None and selector.state == TrackingState.WAITING_USER_SELECT
    print(f"[PASS] Khuất quá 2.0s -> Chuyển sang {selector.state.value} (Đợi người dùng chọn)")

    # Click chuột chọn người #2
    chosen = selector.select_by_point(60, 200, [p1, p2], now=3.0)
    assert chosen == 2 and selector.state == TrackingState.LOCKED
    print(f"[PASS] Click chuột chọn mục tiêu -> Đã khóa ID #{chosen}")
    print("=== Module selector.py hoạt động tốt! ===")
