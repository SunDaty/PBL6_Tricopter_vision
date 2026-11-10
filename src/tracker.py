# src/tracker.py
from typing import Optional

from controller import ControlCommand
from detector import DetectedPerson
from distance_tracker import DistanceTracker
from selector import TargetSelector, TrackingState
from yaw_tracker import YawOnlyTracker


class YawTracker:
    """
    Bộ Tracker TỔNG HỢP dùng trong vòng lặp điều khiển chung:
    Kết hợp 3 thành phần con đã tách riêng:
    1. TargetSelector (selector.py): Lựa chọn và khóa ID mục tiêu.
    2. YawOnlyTracker (yaw_tracker.py): Điều khiển góc hướng Yaw (deg/s).
    3. DistanceTracker (distance_tracker.py): Điều khiển tiến/lùi Vx (m/s) có Deadband pixel
       và tự động chốt cự ly chuẩn khi bắt đầu khóa mục tiêu.
    """
    def __init__(
        self,
        frame_width: int = 640,
        frame_height: int = 480,
        yaw_kp: float = 0.08,
        yaw_kd: float = 0.01,
        yaw_deadband: float = 15.0,
        max_yaw_rate: float = 25.0,
        enable_distance_tracking: bool = False,
        initial_ref_height_px: Optional[float] = None,
        dist_kp: float = 0.015,
        dist_kd: float = 0.003,
        dist_deadband_px: float = 12.0,
        max_vx: float = 1.2,
        lost_timeout: float = 2.0,
        center_weight: float = 0.6,
        area_weight: float = 0.4,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.center_x = frame_width / 2.0
        self.yaw_deadband = yaw_deadband
        self.dist_deadband_px = dist_deadband_px
        self.enable_distance_tracking = enable_distance_tracking

        # 1. Bộ lựa chọn & khóa mục tiêu độc lập
        self.selector = TargetSelector(
            frame_width=frame_width,
            frame_height=frame_height,
            lost_timeout=lost_timeout,
            center_weight=center_weight,
            area_weight=area_weight,
        )

        # 2. Bộ điều khiển góc Yaw độc lập
        self.yaw_tracker = YawOnlyTracker(
            frame_width=frame_width,
            yaw_kp=yaw_kp,
            yaw_kd=yaw_kd,
            yaw_deadband=yaw_deadband,
            max_yaw_rate=max_yaw_rate,
        )

        # 3. Bộ điều khiển khoảng cách Tiến/Lùi độc lập (có Deadband pixel & Auto-Reference)
        self.distance_tracker = DistanceTracker(
            frame_height=frame_height,
            initial_ref_height_px=initial_ref_height_px,
            dist_kp=dist_kp,
            dist_kd=dist_kd,
            dist_deadband_px=dist_deadband_px,
            max_vx=max_vx,
        )

    @property
    def state(self) -> TrackingState:
        return self.selector.state

    @property
    def locked_id(self) -> Optional[int]:
        return self.selector.locked_id

    @property
    def lost_timeout(self) -> float:
        return self.selector.lost_timeout

    @property
    def ref_height_px(self) -> Optional[float]:
        return self.distance_tracker.ref_height_px

    @property
    def current_height_px(self) -> Optional[float]:
        return self.distance_tracker.current_height_px

    @property
    def error_height_px(self) -> Optional[float]:
        return self.distance_tracker.last_error_height_px

    def reset_controllers(self, keep_distance_ref: bool = True):
        """Xóa trạng thái tích phân/đạo hàm của bộ điều khiển Yaw và Distance."""
        self.yaw_tracker.reset()
        self.distance_tracker.reset(keep_reference=keep_distance_ref)

    def reset(self):
        """Khôi phục toàn bộ trạng thái về tự động tìm kiếm ban đầu."""
        self.reset_controllers(keep_distance_ref=False)
        self.selector.reset()

    def unlock_target(self):
        """Hủy khóa mục tiêu hiện tại và chuyển sang trạng thái Hover chờ người dùng chọn."""
        self.reset_controllers(keep_distance_ref=False)
        self.selector.unlock()

    def trigger_auto_select(self):
        """Kích hoạt chế độ tự động chọn mục tiêu tối ưu nhất."""
        self.reset_controllers(keep_distance_ref=False)
        self.selector.trigger_auto_select()

    def toggle_distance_tracking(self) -> bool:
        """
        Bật/tắt nhanh chế độ tiến/lùi ngay trong lúc chạy.
        Khi bật lên sẽ tự động lấy khoảng cách hiện tại của mục tiêu làm chuẩn để không bị trôi.
        """
        self.enable_distance_tracking = not self.enable_distance_tracking
        self.distance_tracker.reset(keep_reference=False)
        return self.enable_distance_tracking

    def recalibrate_distance(self, active_target: Optional[DetectedPerson] = None) -> Optional[float]:
        """Chốt lại chiều cao hiện tại của mục tiêu làm mốc cự ly chuẩn mới."""
        bbox = active_target.bbox if active_target is not None else self.selector.last_known_bbox
        if bbox is not None:
            return self.distance_tracker.recalibrate_from_bbox(bbox)
        self.distance_tracker.reset(keep_reference=False)
        return None

    def adjust_reference_height(self, delta_px: float) -> Optional[float]:
        """Tăng/giảm chiều cao chuẩn mong muốn bằng phím +/-."""
        return self.distance_tracker.adjust_reference_height(delta_px)

    def lock_target_id(self, track_id: int, now: Optional[float] = None):
        """Khóa trực tiếp vào một track_id cụ thể."""
        if self.selector.lock_id(track_id, now=now):
            self.reset_controllers(keep_distance_ref=False)

    def select_by_point(
        self,
        x: int,
        y: int,
        detections: list[DetectedPerson],
        now: Optional[float] = None,
    ) -> Optional[int]:
        """Chọn mục tiêu tại tọa độ click chuột (x, y) và chốt luôn khoảng cách chuẩn của người đó."""
        prev_id = self.selector.locked_id
        chosen_id = self.selector.select_by_point(x, y, detections, now=now)
        if chosen_id is not None and chosen_id != prev_id:
            self.reset_controllers(keep_distance_ref=False)
            if self.selector.last_known_bbox is not None:
                self.distance_tracker.recalibrate_from_bbox(self.selector.last_known_bbox)
        return chosen_id

    def cycle_target(
        self,
        detections: list[DetectedPerson],
        now: Optional[float] = None,
    ) -> Optional[int]:
        """Chuyển đổi luân phiên sang người tiếp theo trong khung hình."""
        prev_id = self.selector.locked_id
        chosen_id = self.selector.cycle_target(detections, now=now)
        if chosen_id is not None and chosen_id != prev_id:
            self.reset_controllers(keep_distance_ref=False)
            if self.selector.last_known_bbox is not None:
                self.distance_tracker.recalibrate_from_bbox(self.selector.last_known_bbox)
        return chosen_id

    def get_occlusion_elapsed(self, now: Optional[float] = None) -> float:
        """Thời gian (giây) kể từ lần cuối nhìn thấy mục tiêu đang khóa."""
        return self.selector.get_occlusion_elapsed(now=now)

    def update_targets(
        self,
        detections: list[DetectedPerson],
        now: Optional[float] = None,
    ) -> tuple[ControlCommand, Optional[DetectedPerson]]:
        """
        Cập nhật danh sách người phát hiện được qua TargetSelector và tính lệnh điều khiển
        tổng hợp (Yaw + Tịnh tiến lồng nhau).
        """
        prev_id = self.selector.locked_id
        target, should_reset_pid = self.selector.update(detections, now=now)

        if target is not None and target.track_id != prev_id:
            # Khi mới khóa vào một ID mới -> chốt chiều cao của ID đó làm mốc chuẩn
            self.reset_controllers(keep_distance_ref=False)
            self.distance_tracker.recalibrate_from_bbox(target.bbox)
        elif should_reset_pid:
            # Khi bị khuất ngắn hạn -> giữ nguyên mốc ref_height_px nhưng xóa tích phân/đạo hàm PID
            self.reset_controllers(keep_distance_ref=(self.selector.locked_id is not None))

        if target is None:
            return ControlCommand(), None

        return self.compute_command(target.bbox, now=now), target

    def compute_command(
        self,
        bbox: Optional[tuple[int, int, int, int]],
        now: Optional[float] = None,
    ) -> ControlCommand:
        """
        Tính toán lệnh điều khiển tổng hợp (vx, vy, vz, yaw_rate) bằng cách gọi
        YawOnlyTracker và DistanceTracker.
        """
        if bbox is None:
            self.reset_controllers(keep_distance_ref=True)
            return ControlCommand()

        yaw_rate = self.yaw_tracker.compute_yaw_rate(bbox, now=now)

        vx = 0.0
        if self.enable_distance_tracking:
            vx = self.distance_tracker.compute_vx(bbox, now=now)
        else:
            # Vẫn cập nhật chiều cao đo đạc để hiển thị trên HUD ngay cả khi đang tắt Vx
            if self.distance_tracker.ref_height_px is None:
                self.distance_tracker.recalibrate_from_bbox(bbox)
            else:
                h = self.distance_tracker.compute_effective_height_px(bbox)
                if h is not None:
                    self.distance_tracker._last_error_height_px = (
                        self.distance_tracker.ref_height_px - h
                    )

        return ControlCommand(vx=vx, vy=0.0, vz=0.0, yaw_rate=yaw_rate)

    def compute_yaw_rate(self, bbox: Optional[tuple[int, int, int, int]]) -> float:
        """Chỉ trả về tốc độ góc yaw (deg/s) thông qua YawOnlyTracker."""
        return self.yaw_tracker.compute_yaw_rate(bbox)


# Alias rõ nghĩa cho bộ Tracker tổng hợp
CombinedTracker = YawTracker


if __name__ == "__main__":
    print("=== Kiểm thử độc lập Module 5: tracker.py (CombinedTracker: Yaw + Distance) ===")
    tracker = CombinedTracker(enable_distance_tracking=True, lost_timeout=2.0)

    # 1. Lần đầu khóa mục tiêu (cao 140px) -> Tự lấy 140px làm mốc chuẩn, vx = 0.0 (Không bị trôi!)
    p1_init = DetectedPerson(track_id=1, bbox=(400, 170, 480, 310), conf=0.92)
    cmd_init, active = tracker.update_targets([p1_init], now=0.0)
    assert active is not None and cmd_init.vx == 0.0 and cmd_init.yaw_rate > 0
    print(
        f"[PASS] 1. Vừa khóa ID #{active.track_id} (ref_h={tracker.ref_height_px:.0f}px): "
        f"yaw_rate={cmd_init.yaw_rate:+.2f} deg/s, vx={cmd_init.vx:+.2f} m/s (Không bị trôi!)"
    )

    # 2. Mục tiêu đi ra xa (cao giảm xuống 90px) -> Tiến theo (vx > 0)
    p1_far = DetectedPerson(track_id=1, bbox=(400, 195, 480, 285), conf=0.92)
    cmd_far, _ = tracker.update_targets([p1_far], now=0.5)
    assert cmd_far.vx > 0
    print(
        f"[PASS] 2. Mục tiêu đi xa (h={tracker.current_height_px:.0f}px): "
        f"yaw_rate={cmd_far.yaw_rate:+.2f} deg/s, vx={cmd_far.vx:+.2f} m/s (Tiến tới)"
    )

    # 3. Mục tiêu tiến lại gần (cao tăng lên 210px) -> Lùi lại (vx < 0)
    p1_close = DetectedPerson(track_id=1, bbox=(380, 135, 500, 345), conf=0.95)
    cmd_close, _ = tracker.update_targets([p1_close], now=1.0)
    assert cmd_close.vx < 0
    print(
        f"[PASS] 3. Mục tiêu lại gần (h={tracker.current_height_px:.0f}px): "
        f"yaw_rate={cmd_close.yaw_rate:+.2f} deg/s, vx={cmd_close.vx:+.2f} m/s (Lùi lại)"
    )
    print("=== Module tracker.py hoạt động tốt! ===")
