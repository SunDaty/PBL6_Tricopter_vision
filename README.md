# Tricopter Person Detection & Tracking (Gazebo SITL & Raspberry Pi 5 + Pixhawk 6C)

Hệ thống **nhận diện và bám theo mục tiêu (con người)** sử dụng **YOLOv8n + ByteTrack** và **ArduPilot**, thiết kế tinh gọn theo kiến trúc module hóa để chuyển đổi trực tiếp từ **Gazebo SITL** sang **Raspberry Pi 5 (Camera Module v3) + Pixhawk 6C**.

---

## 1. Cấu trúc Module Tinh Gọn (6 Module Độc lập)

```text
pbl6_tricopter_vision/
├── models/
│   └── yolov8n.pt              # Trọng số mô hình YOLOv8 Nano
├── src/
│   ├── controller.py           # [Module 1] Bộ điều khiển PID/PD có Soft Deadband & ControlCommand
│   ├── camera.py               # [Module 2] Trừu tượng hóa Camera (GazeboCamera & RPi5 PiCameraV3)
│   ├── mavlink_client.py       # [Module 3] Giao tiếp MAVLink (SITL UDP & Pixhawk 6C UART)
│   ├── detector.py             # [Module 4] Nhận diện & định danh người (YOLOv8n + ByteTrack)
│   ├── tracker.py              # [Module 5] Chọn mục tiêu (Click chuột) + Điều khiển Yaw & Tiến/Lùi
│   ├── visualizer.py           # [Module 6] Vẽ HUD trực quan & xuất log Telemetry
│   └── main_tracker.py         # Chương trình điều phối chính (--mode sim | --mode real)
├── requirements.txt            # Các thư viện Python phụ thuộc
└── README.md                   # Tài liệu hướng dẫn
```

---

## 2. Cơ chế Điều khiển Bám Mục Tiêu (`src/tracker.py`)

1. **Chọn mục tiêu trực tiếp bằng Click Chuột Trái**:
   * Khi chưa chọn mục tiêu, drone đứng yên giữ vị trí (`Hover`: `vx = 0`, `yaw_rate = 0`).
   * Khi **Click chuột trái** vào một người trên cửa sổ Camera, hệ thống khóa `track_id` của người đó, đồng thời tự động chốt chiều cao khung bao hiện tại làm mốc chuẩn `ref_height_px` (`vx = 0.0 m/s`, không bị tự trôi).
2. **Điều khiển góc hướng Yaw (`yaw_rate`)**:
   * Giữ mục tiêu tại tâm ngang khung hình với vùng chết mềm `±15px`.
3. **Điều khiển khoảng cách Tiến/Lùi (`vx`) theo Tỉ lệ Chiều cao Chuẩn hóa**:
   * Sử dụng sai số tỉ lệ chuẩn hóa: `error_ratio = (ref_height_px - current_h) / ref_height_px` kèm vùng chết mềm `±8%` (`dist_deadband_ratio = 0.08`), giúp độ nhạy phản ứng nhất quán ở mọi khoảng cách.
   * **Người đi ra xa** (`current_h < 0.92 * ref_height_px`) $\rightarrow$ **`vx > 0` (Drone tiến theo)**.
   * **Người tiến lại gần** (`current_h > 1.08 * ref_height_px`) $\rightarrow$ **`vx < 0` (Drone lùi lại)**.
   * **Mất dấu quá `2.0s`** (`lost_timeout`) $\rightarrow$ Tự động hủy khóa và dừng Hover an toàn.

---

## 3. Kiểm thử Độc lập Từng Module

```bash
python3 src/controller.py
python3 src/camera.py --test-mock
python3 src/mavlink_client.py --test-mock
python3 src/detector.py
python3 src/tracker.py
python3 src/visualizer.py
```

---

## 4. Hướng dẫn Vận hành

### 4.1. Chạy trong Mô phỏng Gazebo + ArduPilot SITL
```bash
python3 src/main_tracker.py --mode sim
```

### 4.2. Chạy trên Thực tế: Raspberry Pi 5 (Camera v3) + Pixhawk 6C
```bash
python3 src/main_tracker.py --mode real --mavlink /dev/ttyAMA0 --baud 921600 --imgsz 320
```

### 4.3. Thao tác trên Cửa sổ Camera

| Thao tác | Chức năng |
| :--- | :--- |
| **Click Chuột Trái** vào người | Chọn khóa mục tiêu và tự động bay bám theo (Yaw + Tiến/Lùi) |
| Phím **`Q`** | Dừng drone (Hover) và thoát chương trình |
