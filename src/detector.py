# src/detector.py
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional
import numpy as np
from ultralytics import YOLO


@dataclass
class DetectedPerson:
    """Thông tin một người được phát hiện và gán ID bởi bộ Multi-Object Tracker."""
    track_id: int
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    conf: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    def contains_point(self, x: int, y: int) -> bool:
        x1, y1, x2, y2 = self.bbox
        return x1 <= x <= x2 and y1 <= y <= y2


class HumanDetector:
    """
    Module nhận diện và định danh người (YOLOv8 + ByteTrack).
    - Hỗ trợ tùy chỉnh imgsz (640 trên PC, 320 hoặc 416 trên Raspberry Pi 5 để tăng FPS).
    - Hỗ trợ nạp cả model PyTorch (.pt) lẫn model NCNN/ONNX trên Raspberry Pi 5.
    """
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_thresh: float = 0.5,
        imgsz: int = 640,
        device: Optional[str] = None,
        tracker_config: str = "bytetrack.yaml",
    ):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz
        self.device = device
        self.tracker_config = tracker_config

    def detect_and_track(self, frame: np.ndarray) -> list[DetectedPerson]:
        """
        Chạy YOLOv8 kèm ByteTrack để phát hiện tất cả người (class 0) và duy trì track_id.
        Trả về danh sách DetectedPerson trong khung hình hiện tại.
        """
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_config,
            classes=[0],
            conf=self.conf_thresh,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        detections: list[DetectedPerson] = []
        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0]) if box.conf is not None else 0.0

                if box.id is not None:
                    track_id = int(box.id[0])
                else:
                    track_id = -(idx + 1)

                detections.append(
                    DetectedPerson(
                        track_id=track_id,
                        bbox=(x1, y1, x2, y2),
                        conf=conf,
                    )
                )

        return detections

    def detect(self, frame: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        """Hàm tương thích ngược: trả về bbox của người có diện tích lớn nhất."""
        detections = self.detect_and_track(frame)
        if not detections:
            return None
        best = max(detections, key=lambda d: d.area)
        return best.bbox


if __name__ == "__main__":
    print("=== Kiểm thử độc lập Module 8: detector.py (HumanDetector YOLOv8 + ByteTrack) ===")
    model_file = Path(__file__).resolve().parent / "models" / "yolov8n.pt"
    detector = HumanDetector(model_path=str(model_file), imgsz=320)

    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    t0 = time.monotonic()
    dets = detector.detect_and_track(dummy_img)
    latency_ms = (time.monotonic() - t0) * 1000.0
    assert isinstance(dets, list)
    print(f"[PASS] Suy luận YOLOv8n + ByteTrack thành công (latency={latency_ms:.1f} ms, detections={len(dets)})")
    print("=== Module detector.py hoạt động tốt! ===")
