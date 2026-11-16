# src/detector.py
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional
import numpy as np
from ultralytics import YOLO


@dataclass
class DetectedPerson:
    """Thông tin một người được phát hiện và gán ID bởi ByteTrack."""
    track_id: int
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    conf: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def width(self) -> float:
        return max(1.0, float(self.bbox[2] - self.bbox[0]))

    @property
    def height(self) -> float:
        return max(1.0, float(self.bbox[3] - self.bbox[1]))

    def contains_point(self, x: int, y: int) -> bool:
        x1, y1, x2, y2 = self.bbox
        return x1 <= x <= x2 and y1 <= y <= y2


class HumanDetector:
    """
    Module nhận diện và định danh người (YOLOv8n + ByteTrack).
    - Hỗ trợ tùy chỉnh imgsz (640 trên PC Sim, 320 trên Raspberry Pi 5 để đạt FPS cao).
    - Đặt conf_thresh=0.3 để ByteTrack tận dụng được vòng ghép thứ 2 (low-score association)
      nhằm duy trì track_id bền vững khi camera quay hoặc người xoay ngang.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_thresh: float = 0.3,
        imgsz: int = 640,
        device: Optional[str] = None,
        tracker_config: str = "bytetrack.yaml",
    ):
        if model_path is None:
            model_path = str(Path(__file__).resolve().parent.parent / "models" / "yolov8n.pt")
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz
        self.device = device
        self.tracker_config = tracker_config

    def detect_and_track(self, frame: np.ndarray) -> list[DetectedPerson]:
        """
        Chạy YOLOv8 kèm ByteTrack để phát hiện người (class 0) và duy trì track_id.
        Chỉ trả về các mục tiêu đã được ByteTrack cấp ID hợp lệ.
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

            for box in boxes:
                if box.id is None:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0]) if box.conf is not None else 0.0
                track_id = int(box.id[0])

                detections.append(
                    DetectedPerson(
                        track_id=track_id,
                        bbox=(x1, y1, x2, y2),
                        conf=conf,
                    )
                )

        return detections


if __name__ == "__main__":
    print("=== Kiểm thử độc lập Module 4: detector.py (HumanDetector YOLOv8 + ByteTrack) ===")
    detector = HumanDetector(imgsz=320)

    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    t0 = time.monotonic()
    dets = detector.detect_and_track(dummy_img)
    latency_ms = (time.monotonic() - t0) * 1000.0
    assert isinstance(dets, list)
    print(f"[PASS] Suy luận YOLOv8n + ByteTrack thành công (latency={latency_ms:.1f} ms, detections={len(dets)})")
    print("=== Module detector.py hoạt động tốt! ===")
