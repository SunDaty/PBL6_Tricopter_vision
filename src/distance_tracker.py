# src/distance_tracker.py
from typing import Optional
from controller import PIDController


class DistanceTracker:
    """
    Module điều khiển ĐỘC LẬP dành riêng cho chuyển động Tịnh tiến Tiến/Lùi (vx).
    Khắc phục triệt để hiện tượng tự trôi 1 hướng và không lùi lại:
    1. Tự động lấy chiều cao khung bao lúc bắt đầu khóa mục tiêu làm mốc chuẩn (ref_height_px),
       giúp tại vị trí ban đầu sai số = 0 -> vx = 0 (Drone đứng yên, không bị tự trôi).
    2. Có vùng chết Tiến/Lùi rõ ràng theo đơn vị pixel (dist_deadband_px = ±12px).
    3. Bù cắt mép dưới khung hình (Bottom-Edge Clipping): Khi người tiến lại gần bị cắt chân ở
       đáy ảnh (y2 chạm mép dưới), sử dụng chiều rộng khung bao nhân tỉ lệ khung người để suy ra
       chiều cao thực tế, đảm bảo drone luôn nhận biết người đang tiến sát và lùi lại (vx < 0).
    """
    def __init__(
        self,
        frame_height: int = 480,
        initial_ref_height_px: Optional[float] = None,
        dist_kp: float = 0.015,
        dist_kd: float = 0.003,
        dist_deadband_px: float = 12.0,
        max_vx: float = 1.2,
        height_ema_alpha: float = 0.4,
    ):
        self.frame_height = frame_height
        self.default_ref_height_px = initial_ref_height_px
        self.ref_height_px: Optional[float] = initial_ref_height_px
        self.dist_deadband_px = dist_deadband_px
        self.max_vx = max_vx
        self.height_ema_alpha = height_ema_alpha

        self._ref_aspect_ratio: float = 2.4  # Tỉ lệ cao/rộng mặc định của người đứng
        self._smoothed_height_px: Optional[float] = None
        self._last_error_height_px: Optional[float] = None

        # Bộ điều khiển PD theo đơn vị pixel (đầu vào: error_height_px, đầu ra: vx m/s)
        self.pid = PIDController(
            kp=dist_kp,
            kd=dist_kd,
            deadband=dist_deadband_px,
            output_limits=(-max_vx, max_vx),
        )

    def reset(self, keep_reference: bool = False):
        """
        Xóa trạng thái bộ điều khiển khoảng cách.
        Nếu keep_reference=False, sẽ tự động lấy lại chiều cao chuẩn ở frame có người tiếp theo.
        """
        self.pid.reset()
        self._smoothed_height_px = None
        self._last_error_height_px = None
        if not keep_reference:
            self.ref_height_px = self.default_ref_height_px

    def recalibrate_from_bbox(self, bbox: tuple[int, int, int, int]) -> float:
        """Lấy chiều cao khung bao hiện tại làm mốc khoảng cách chuẩn mới (ref_height_px)."""
        x1, y1, x2, y2 = bbox
        raw_w = max(1.0, float(x2 - x1))
        raw_h = max(1.0, float(y2 - y1))

        # Nếu người chưa bị cắt ở mép dưới/trên, cập nhật tỉ lệ cao/rộng chuẩn
        if y1 > 8 and y2 < (self.frame_height - 8):
            self._ref_aspect_ratio = max(1.2, min(4.0, raw_h / raw_w))

        self.ref_height_px = max(50.0, min(float(self.frame_height * 0.80), raw_h))
        self._smoothed_height_px = self.ref_height_px
        self._last_error_height_px = 0.0
        self.pid.reset()
        return self.ref_height_px

    def adjust_reference_height(self, delta_px: float) -> Optional[float]:
        """
        Điều chỉnh mốc chiều cao chuẩn bằng phím +/-:
        - delta_px > 0: Muốn giữ khoảng cách gần người hơn (khung chuẩn lớn hơn).
        - delta_px < 0: Muốn giữ khoảng cách xa người hơn (khung chuẩn nhỏ hơn).
        """
        if self.ref_height_px is None:
            self.ref_height_px = 140.0
        self.ref_height_px = max(
            50.0,
            min(float(self.frame_height * 0.80), self.ref_height_px + delta_px),
        )
        self.pid.reset()
        return self.ref_height_px

    def compute_effective_height_px(
        self, bbox: Optional[tuple[int, int, int, int]]
    ) -> Optional[float]:
        """
        Tính chiều cao hiệu dụng (pixel) của người:
        - Tự động bù chiều cao thông qua bề ngang (width * aspect_ratio) nếu người tiến quá gần
          làm chân/đầu bị cắt ở mép khung hình (y2 >= frame_height - 8 hoặc y1 <= 8).
        """
        if bbox is None or self.frame_height <= 0:
            return None

        x1, y1, x2, y2 = bbox
        raw_w = max(1.0, float(x2 - x1))
        raw_h = max(1.0, float(y2 - y1))

        is_clipped = (y2 >= self.frame_height - 8) or (y1 <= 8)
        if is_clipped:
            estimated_h = raw_w * self._ref_aspect_ratio
            effective_h = max(raw_h, estimated_h)
        else:
            # Cập nhật nhẹ tỉ lệ khung hình khi nhìn thấy toàn thân
            aspect = max(1.2, min(4.0, raw_h / raw_w))
            self._ref_aspect_ratio = 0.9 * self._ref_aspect_ratio + 0.1 * aspect
            effective_h = raw_h

        if self._smoothed_height_px is None:
            self._smoothed_height_px = effective_h
        else:
            self._smoothed_height_px = (
                self.height_ema_alpha * effective_h
                + (1.0 - self.height_ema_alpha) * self._smoothed_height_px
            )
        return self._smoothed_height_px

    @property
    def current_height_px(self) -> Optional[float]:
        return self._smoothed_height_px

    @property
    def last_error_height_px(self) -> Optional[float]:
        return self._last_error_height_px

    def get_height_ratio(self, bbox: Optional[tuple[int, int, int, int]]) -> Optional[float]:
        """Tỉ lệ chiều cao khung bao so với chiều cao ảnh [0.0, 1.0]."""
        if bbox is None or self.frame_height <= 0:
            return None
        _, y1, _, y2 = bbox
        return max(0.0, (y2 - y1) / float(self.frame_height))

    def compute_vx(
        self,
        bbox: Optional[tuple[int, int, int, int]],
        now: Optional[float] = None,
    ) -> float:
        """
        Tính vận tốc tiến/lùi vx (m/s) từ bounding box:
        - Nếu chưa có ref_height_px: tự động lấy chiều cao hiện tại làm chuẩn -> vx = 0.0 (Không tự trôi).
        - Nếu |ref_height_px - current_h| < dist_deadband_px -> vx = 0.0 (Trong vùng Deadband).
        - Nếu người đi xa (current_h < ref_height_px - db) -> error > 0 -> vx > 0 (Tiến theo).
        - Nếu người lại gần (current_h > ref_height_px + db) -> error < 0 -> vx < 0 (Lùi lại).
        """
        if bbox is None:
            self.pid.reset()
            self._last_error_height_px = None
            return 0.0

        # Lần đầu tiên nhìn thấy mục tiêu đã khóa -> chốt luôn cự ly hiện tại làm chuẩn
        if self.ref_height_px is None:
            self.recalibrate_from_bbox(bbox)
            return 0.0

        current_h = self.compute_effective_height_px(bbox)
        if current_h is None:
            self.pid.reset()
            self._last_error_height_px = None
            return 0.0

        # Sai số chiều cao (pixel):
        # > 0 khi người nhỏ hơn chuẩn (ở xa -> cần tiến vx > 0)
        # < 0 khi người lớn hơn chuẩn (lại gần -> cần lùi vx < 0)
        error_h_px = self.ref_height_px - current_h
        self._last_error_height_px = error_h_px
        return self.pid.compute(error_h_px, now=now)


if __name__ == "__main__":
    print("=== Kiểm thử độc lập Module 4: distance_tracker.py (DistanceTracker) ===")
    dist_ctrl = DistanceTracker(frame_height=480, dist_kp=0.015, dist_deadband_px=12.0, height_ema_alpha=1.0)

    # 1. Khung hình đầu tiên khi vừa khóa mục tiêu (cao 130px) -> Tự lấy 130px làm chuẩn, vx = 0.0 (Không bị trôi!)
    vx_init = dist_ctrl.compute_vx((290, 175, 350, 305), now=0.0)
    assert vx_init == 0.0 and dist_ctrl.ref_height_px == 130.0
    print(f"[PASS] 1. Tự chốt mốc chuẩn ban đầu: ref_h={dist_ctrl.ref_height_px:.0f}px -> vx={vx_init:+.2f} m/s (Không bị trôi)")

    # 2. Người nhúc nhích nhẹ trong vùng Deadband (cao 138px, lệch -8px nằm trong ±12px) -> vx = 0.0
    vx_db = dist_ctrl.compute_vx((288, 171, 352, 309), now=0.5)
    assert vx_db == 0.0
    print(f"[PASS] 2. Trong vùng Deadband (±12px): h=138px (err={dist_ctrl.last_error_height_px:+.1f}px) -> vx={vx_db:+.2f} m/s")

    # 3. Người đi ra xa (cao giảm xuống 90px < 130 - 12px) -> Drone tiến theo (vx > 0)
    vx_far = dist_ctrl.compute_vx((300, 195, 340, 285), now=1.0)
    assert vx_far > 0
    print(f"[PASS] 3. Người đi xa (h=90px, err={dist_ctrl.last_error_height_px:+.1f}px) -> vx={vx_far:+.2f} m/s (Tiến tới)")

    # 4. Người bước lại gần (cao tăng lên 180px > 130 + 12px) -> Drone lùi lại ngay (vx < 0)
    dist_ctrl.pid.reset()
    vx_close = dist_ctrl.compute_vx((275, 150, 365, 330), now=2.0)
    assert vx_close < 0
    print(f"[PASS] 4. Người tiến lại gần (h=180px, err={dist_ctrl.last_error_height_px:+.1f}px) -> vx={vx_close:+.2f} m/s (Lùi lại)")

    # 5. Người tiến quá sát làm chân bị cắt ở đáy ảnh (y2 = 479, rộng tăng lên 140px) -> Vẫn lùi lại mạnh (vx < 0)
    vx_clipped = dist_ctrl.compute_vx((250, 320, 390, 479), now=3.0)
    assert vx_clipped < 0
    print(f"[PASS] 5. Người tiến sát bị cắt mép dưới (effective_h={dist_ctrl.current_height_px:.0f}px) -> vx={vx_clipped:+.2f} m/s (Lùi lại mạnh)")
    print("=== Module distance_tracker.py hoạt động tốt! ===")
