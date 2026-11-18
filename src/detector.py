# src/detector.py
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional
import cv2
import ncnn
import numpy as np
import supervision as sv

# Giới hạn số luồng để tránh tranh chấp CPU và chống sụt áp (Brownout) trên Raspberry Pi 5
cv2.setNumThreads(1)
DEFAULT_NCNN_THREADS = 2
REG_MAX = 16


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


def _make_anchors(input_size: int = 640, strides: tuple[int, ...] = (8, 16, 32)):
    anchor_points = []
    stride_tensor = []
    for stride in strides:
        grid_h = input_size // stride
        grid_w = input_size // stride
        yv, xv = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing="ij")
        points = np.stack((xv, yv), axis=-1).reshape(-1, 2) + 0.5
        anchor_points.append(points)
        stride_tensor.append(np.full((points.shape[0], 1), stride, dtype=np.float32))
    return np.concatenate(anchor_points, axis=0), np.concatenate(stride_tensor, axis=0)


def _resolve_ncnn_files(model_path: Optional[str]) -> tuple[Path, Path]:
    """Tìm cặp file .param và .bin từ đường dẫn thư mục hoặc file."""
    if model_path is None:
        base_dir = Path(__file__).resolve().parent.parent / "models" / "yolov8n_ncnn_model"
    else:
        base_dir = Path(model_path)

    if base_dir.is_file():
        if base_dir.suffix == ".param":
            return base_dir, base_dir.with_suffix(".bin")
        if base_dir.suffix == ".bin":
            return base_dir.with_suffix(".param"), base_dir

    candidates = [
        (base_dir / "model.ncnn.param", base_dir / "model.ncnn.bin"),
        (base_dir / "yolov8n.param", base_dir / "yolov8n.bin"),
    ]
    for param_file, bin_file in candidates:
        if param_file.exists() and bin_file.exists():
            return param_file, bin_file

    raise FileNotFoundError(
        f"Không tìm thấy cặp file NCNN (.param / .bin) tại: {base_dir}"
    )


class HumanDetector:
    """
    Module nhận diện và định danh người (YOLOv8n NCNN + ByteTrack).
    - Dùng trực tiếp ncnn.Net với cấu hình chống sụt áp trên Raspberry Pi 5 (num_threads=2, lightmode, fp16).
    - Hỗ trợ giải mã cả 2 chuẩn đầu ra NCNN: 144 kênh (Qengineering DFL) và 84 kênh (Ultralytics PNNX).
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_thresh: float = 0.45,
        nms_thresh: float = 0.45,
        imgsz: int = 640,
        num_threads: int = DEFAULT_NCNN_THREADS,
    ):
        self.param_path, self.bin_path = _resolve_ncnn_files(model_path)
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.imgsz = imgsz
        self.num_threads = num_threads

        with open(self.param_path, "r", encoding="utf-8", errors="ignore") as f:
            first_line = f.readline().strip()
            if first_line != "7767517":
                raise ValueError(
                    f"File param không hợp lệ ({self.param_path}, dòng đầu: {first_line})"
                )
            param_content = f.read()

        self.input_blob = "in0" if "in0" in param_content else "images"
        self.output_blob = (
            "out0"
            if "out0" in param_content
            else ("output0" if "output0" in param_content else "output")
        )

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.lightmode = True
        self.net.opt.use_fp16_packed = True
        self.net.opt.use_fp16_storage = True
        self.net.opt.use_fp16_arithmetic = True
        self.net.opt.num_threads = self.num_threads

        if (
            self.net.load_param(str(self.param_path)) != 0
            or self.net.load_model(str(self.bin_path)) != 0
        ):
            raise RuntimeError(
                f"Không thể nạp mô hình NCNN: {self.param_path} / {self.bin_path}"
            )

        self._anchor_points, self._stride_tensor = _make_anchors(self.imgsz)
        self._dfl_proj = np.arange(REG_MAX, dtype=np.float32)
        self.tracker = sv.ByteTrack()

        print(
            f"[*] [HumanDetector] Đã nạp NCNN ({self.param_path.name}, "
            f"input='{self.input_blob}', output='{self.output_blob}', "
            f"threads={self.num_threads}, imgsz={self.imgsz})"
        )

    def _infer_person_detections(self, frame: np.ndarray) -> sv.Detections:
        img_h, img_w = frame.shape[:2]

        # Tiền xử lý Letterbox về (imgsz, imgsz)
        scale = min(self.imgsz / img_w, self.imgsz / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        resized = cv2.resize(frame, (new_w, new_h))
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        pad_w = (self.imgsz - new_w) // 2
        pad_h = (self.imgsz - new_h) // 2
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        mat_in = ncnn.Mat.from_pixels(
            canvas, ncnn.Mat.PixelType.PIXEL_BGR2RGB, self.imgsz, self.imgsz
        )
        mat_in.substract_mean_normalize([0.0, 0.0, 0.0], [1 / 255.0, 1 / 255.0, 1 / 255.0])

        ex = self.net.create_extractor()
        ex.input(self.input_blob, mat_in)
        ret, mat_out = ex.extract(self.output_blob)
        if ret != 0:
            return sv.Detections.empty()

        out = np.array(mat_out)
        if out.ndim == 3:
            out = out.squeeze(0)
        if out.shape[0] < out.shape[1]:
            out = out.T

        num_channels = out.shape[1]

        if num_channels == 144:
            raw_person_score = out[:, 64]
            if np.min(raw_person_score) < 0.0 or np.max(raw_person_score) > 1.0:
                person_scores = 1.0 / (1.0 + np.exp(-np.clip(raw_person_score, -50.0, 50.0)))
            else:
                person_scores = raw_person_score

            mask = person_scores > self.conf_thresh
            if not np.any(mask):
                return sv.Detections.empty()

            scores = person_scores[mask]
            dfl_boxes = out[mask, :64].reshape(-1, 4, REG_MAX)
            anchors = self._anchor_points[mask]
            strides = self._stride_tensor[mask]

            dfl_exp = np.exp(dfl_boxes - np.max(dfl_boxes, axis=-1, keepdims=True))
            dfl_prob = dfl_exp / np.sum(dfl_exp, axis=-1, keepdims=True)
            dist = np.dot(dfl_prob, self._dfl_proj)

            x1 = (anchors[:, 0] - dist[:, 0]) * strides[:, 0]
            y1 = (anchors[:, 1] - dist[:, 1]) * strides[:, 0]
            x2 = (anchors[:, 0] + dist[:, 2]) * strides[:, 0]
            y2 = (anchors[:, 1] + dist[:, 3]) * strides[:, 0]

        elif num_channels == 84:
            person_scores = out[:, 4]
            mask = person_scores > self.conf_thresh
            if not np.any(mask):
                return sv.Detections.empty()

            valid_preds = out[mask]
            scores = person_scores[mask]
            cx, cy, w, h = valid_preds[:, 0], valid_preds[:, 1], valid_preds[:, 2], valid_preds[:, 3]
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
        else:
            raise ValueError(f"Số kênh đầu ra NCNN không hỗ trợ: {num_channels}")

        # Quy đổi tọa độ về khung hình gốc
        x1 = np.clip((x1 - pad_w) / scale, 0, img_w)
        y1 = np.clip((y1 - pad_h) / scale, 0, img_h)
        x2 = np.clip((x2 - pad_w) / scale, 0, img_w)
        y2 = np.clip((y2 - pad_h) / scale, 0, img_h)

        valid_wh = ((x2 - x1) > 1.0) & ((y2 - y1) > 1.0)
        if not np.any(valid_wh):
            return sv.Detections.empty()

        x1, y1, x2, y2, scores = (
            x1[valid_wh], y1[valid_wh], x2[valid_wh], y2[valid_wh], scores[valid_wh]
        )
        boxes_xywh = np.column_stack((x1, y1, x2 - x1, y2 - y1)).tolist()

        indices = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), self.conf_thresh, self.nms_thresh)
        if len(indices) == 0:
            return sv.Detections.empty()

        indices = np.array(indices).flatten()
        xyxy = np.column_stack((x1[indices], y1[indices], x2[indices], y2[indices])).astype(np.float32)
        confidences = scores[indices].astype(np.float32)
        class_ids = np.zeros(len(indices), dtype=int)

        return sv.Detections(xyxy=xyxy, confidence=confidences, class_id=class_ids)

    def detect_and_track(self, frame: np.ndarray) -> list[DetectedPerson]:
        """
        Chạy YOLOv8n NCNN kèm ByteTrack để phát hiện người (class 0) và duy trì track_id.
        Chỉ trả về các mục tiêu đã được ByteTrack cấp ID hợp lệ.
        """
        sv_detections = self._infer_person_detections(frame)
        tracked = self.tracker.update_with_detections(sv_detections)
        if len(tracked) == 0 or tracked.tracker_id is None:
            return []

        detections: list[DetectedPerson] = []
        for box, track_id, conf in zip(tracked.xyxy, tracked.tracker_id, tracked.confidence):
            x1, y1, x2, y2 = map(int, box)
            detections.append(
                DetectedPerson(
                    track_id=int(track_id),
                    bbox=(x1, y1, x2, y2),
                    conf=float(conf) if conf is not None else 0.0,
                )
            )
        return detections


if __name__ == "__main__":
    print("=== Kiểm thử độc lập Module 4: detector.py (HumanDetector YOLOv8n NCNN + ByteTrack) ===")
    detector = HumanDetector(imgsz=640)

    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    t0 = time.monotonic()
    dets = detector.detect_and_track(dummy_img)
    latency_ms = (time.monotonic() - t0) * 1000.0
    assert isinstance(dets, list)
    print(f"[PASS] Suy luận YOLOv8n NCNN + ByteTrack thành công (latency={latency_ms:.1f} ms, detections={len(dets)})")
    print("=== Module detector.py hoạt động tốt! ===")
