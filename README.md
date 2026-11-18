# Tricopter Person Detection & Tracking (Raspberry Pi 5 + Pixhawk 6C & Gazebo SITL)

Hệ thống **nhận diện và bám theo mục tiêu (con người)** thời gian thực sử dụng **YOLOv8n NCNN + ByteTrack** và **ArduPilot/Pixhawk 6C**, tổ chức theo kiến trúc module phẳng gọn nhẹ trong `src/` kết hợp cấu hình YAML trong `config/` và Web Dashboard giám sát qua trình duyệt.

---

## 1. Cấu trúc Dự án

```text
pbl6_tricopter_vision/
├── main.py                      # Điểm chạy chính (python3 main.py --profile raspberry_pi | simulation | mock)
├── config/
│   ├── common.yaml              # Tham số dùng chung (khung hình, detector, PID, tracker)
│   ├── raspberry_pi.yaml        # Cấu hình RPi 5 (/dev/ttyAMA0 @ 921600, PiCam v3 20fps, Web port 5000)
│   ├── simulation.yaml          # Cấu hình Gazebo + ArduPilot SITL (udpin:0.0.0.0:14550)
│   └── mock.yaml                # Cấu hình kiểm thử nhanh không cần phần cứng
├── models/
│   └── yolov8n_ncnn_model/      # Trọng số YOLOv8n NCNN (model.ncnn.param, model.ncnn.bin)
├── src/
│   ├── config_loader.py         # Nạp & gộp đệ quy common.yaml + profile yaml + tham số CLI
│   ├── controller.py            # Bộ điều khiển PID/PD có Soft Deadband & ControlCommand
│   ├── camera.py                # Nguồn Camera (PiCameraV3, GazeboCamera, MockCamera)
│   ├── mavlink_client.py        # Giao tiếp MAVLink (guided_vel, guided_nogps, rc_override)
│   ├── detector.py              # Nhận diện & định danh người (YOLOv8n NCNN + ByteTrack)
│   ├── tracker.py               # Khóa mục tiêu (Click hoặc Auto-lock) + Điều khiển Yaw & Vx
│   ├── visualizer.py            # Vẽ HUD & xuất log Telemetry
│   ├── web_server.py            # Flask MJPEG Stream (chống sụt áp + Click chọn mục tiêu trên Web)
│   ├── metrics.py               # Đo đạc FPS, độ trễ E2E, overshoot yaw, xuất CSV/report
│   └── main_tracker.py          # Vòng lặp điều phối chính (DroneTrackerApp)
├── logs/                        # Lưu báo cáo & CSV debug (<timestamp>/frames.csv, steps.csv, report.txt)
└── requirements.txt
```

---

## 2. Kiểm thử Độc lập Từng Module

```bash
source /home/dat/drone_tracking/drone_env/bin/activate
cd /home/dat/drone_tracking/pbl6_tricopter_vision

python3 src/config_loader.py
python3 src/web_server.py
python3 src/controller.py
python3 src/camera.py --test-mock
python3 src/camera.py --mode real
python3 src/mavlink_client.py --test-mock
python3 src/detector.py
python3 src/tracker.py
python3 src/visualizer.py
python3 src/metrics.py
```

---

## 3. Hướng dẫn Vận hành

### 3.1. Chạy trên Raspberry Pi 5 (Camera Module v3 + Pixhawk 6C)
```bash
python3 main.py --profile raspberry_pi
```
* Mặc định kết nối UART `/dev/ttyAMA0` ở tốc độ `921600`, Camera v3 `640x480 @ 20fps` (xoay 180°), bật Web Stream tại `http://<IP_Raspberry_Pi>:5000`.
* Có thể chuyển phương thức điều khiển ngay qua CLI nếu cần:
  ```bash
  python3 main.py --profile raspberry_pi --control-method rc_override
  python3 main.py --profile raspberry_pi --control-method guided_nogps
  ```

### 3.2. Chạy trong Mô phỏng Gazebo + ArduPilot SITL
```bash
python3 main.py --profile simulation
```

### 3.3. Chạy thử không cần phần cứng (Mock Camera + Dry-run MAVLink + Web Stream)
```bash
python3 main.py --profile mock --debug
```

### 3.4. Điều khiển trên Web Dashboard & Cửa sổ OpenCV
* **Web Dashboard (`http://<IP>:5000`)**: Chạm/click trực tiếp lên người trong video để khóa mục tiêu hoặc nhấn **Reset / Hover** để hủy khóa. Ảnh JPEG chỉ được nén khi có trình duyệt đang kết nối để tiết kiệm CPU và tránh sụt áp trên Raspberry Pi 5.
* **Cửa sổ OpenCV (khi chạy `--gui` hoặc profile `simulation`)**: Click chuột trái vào người để khóa mục tiêu, nhấn `D` để bật/tắt panel debug, nhấn `Q` để dừng Hover và thoát.

---

## 4. Debug & Đánh giá Hệ thống (`--debug`)

```bash
python3 main.py --profile raspberry_pi --debug
```
Kết quả được lưu vào `logs/<timestamp>/` gồm `frames.csv`, `steps.csv` và `report.txt`.
