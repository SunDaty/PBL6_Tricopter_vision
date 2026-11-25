# Tricopter Person Detection & Tracking (Raspberry Pi 5 + Pixhawk 6C & Gazebo SITL)

Hệ thống **nhận diện và bám theo mục tiêu (con người)** thời gian thực sử dụng **YOLOv8n NCNN + ByteTrack bù chuyển động camera (Ego-Motion)** và **ArduPilot/Pixhawk 6C**, tổ chức theo kiến trúc module phẳng gọn nhẹ trong `src/`, cấu hình YAML phân tầng trong `config/`, Web Dashboard Flask và bộ đo đạc đánh giá tự động.

> **Dành cho AI Agent / Đồng bộ 2 máy (RPi 5 Thực tế ↔ Laptop Mô phỏng):** Xem tệp điều phối **[AGENT_SYNC.md](AGENT_SYNC.md)** trước khi chỉnh sửa hoặc chạy mô phỏng.

---

## 1. Cấu trúc Dự án

```text
pbl6_tricopter_vision/
├── main.py                           # Điểm chạy chính (python3 main.py --profile raspberry_pi | simulation | mock)
├── plot_logs.py                      # Xuất 3 biểu đồ PNG từ logs/<timestamp>/frames.csv (cần matplotlib)
├── AGENT_SYNC.md                     # Tài liệu giao tiếp & bàn giao giữa Agent RPi 5 (thực tế) và Agent Laptop (mô phỏng)
├── rpi5_measured_summary.md          # Bảng tổng hợp toàn bộ 11 phiên đo thực tế trên RPi 5
├── report_best_run_20261119_143256.md# Số liệu báo cáo chuẩn của phiên đo tốt nhất (logs/20261119_143256)
├── report_export_pack.html           # Tệp HTML offline (1 file duy nhất): lọc số liệu, copy Excel/Word & tải 4 đồ thị
├── config/
│   ├── common.yaml                   # Tham số dùng chung (khung hình, camera, detector, tracker, telemetry, loop)
│   ├── raspberry_pi.yaml             # Profile RPi 5 (/dev/ttyAMA0 @ 921600, PiCam v3 20fps, Web port 5000)
│   ├── simulation.yaml               # Profile Gazebo + ArduPilot SITL (udpin:0.0.0.0:14550, GUI)
│   └── mock.yaml                     # Profile kiểm thử nhanh không cần phần cứng
├── models/
│   └── yolov8n_ncnn_model/           # Trọng số YOLOv8n NCNN (model.ncnn.param, model.ncnn.bin)
├── src/
│   ├── config_loader.py              # Nạp & gộp đệ quy common.yaml + profile yaml + tham số CLI
│   ├── ego_motion.py                 # YawHistory (unwrap/nội suy 50Hz) & BearingProjector (chiếu trụ thuận-nghịch)
│   ├── controller.py                 # ControlCommand, PIDController, HysteresisDeadband, SlewRateLimiter
│   ├── camera.py                     # Nguồn Camera (PiCameraV3, GazeboCamera, MockCamera — lazy import)
│   ├── mavlink_client.py             # Giao tiếp MAVLink (guided_vel, guided_nogps, rc_override) + ATTITUDE 50Hz
│   ├── detector.py                   # YOLOv8n NCNN + ByteTrack (conf 0.20 / act 0.45) có bù ego-motion
│   ├── tracker.py                    # Máy trạng thái TRACK/COAST/SEARCH_HOLD, RE-ACQ, bù trễ Yaw & điều khiển Vx
│   ├── visualizer.py                 # Vẽ HUD ('minimal' mặc định không copy/blend, 'full', 'off')
│   ├── web_server.py                 # Flask MJPEG Stream (nén JPEG theo nhu cầu + click chọn mục tiêu trên Web)
│   ├── metrics.py                    # Đo FPS, độ trễ E2E, TIMESYNC, ID quality, YawStepAnalyzer, xuất CSV/report
│   └── main_tracker.py               # Vòng lặp điều phối chính (DroneTrackerApp)
├── tests/                            # Bộ test tự động (ego-motion, tracker states, ByteTrack ego, vòng kín Yaw)
├── tools/
│   └── read_fc_yaw_params.py         # Công cụ chỉ đọc (read-only) tham số trục Yaw & servo đuôi trên Pixhawk 6C
├── logs/                             # Lưu CSV & báo cáo debug (được gitignore)
└── requirements.txt
```

---

## 2. Phân tách Module giữa Thực tế (RPi 5) và Mô phỏng (Laptop)

Nhờ cơ chế **lazy import** trong `src/camera.py`, toàn bộ repo dùng chung giữa 2 máy mà không xung đột thư viện:
- **Dùng chung trên cả 2 máy:** `main.py`, `src/main_tracker.py`, `src/config_loader.py`, `src/ego_motion.py`, `src/detector.py`, `src/tracker.py`, `src/controller.py`, `src/mavlink_client.py`, `src/visualizer.py`, `src/metrics.py`, `plot_logs.py`, `tests/`.
- **Chỉ dùng trên RPi 5 (không cần cài/chạy khi mô phỏng trên Laptop):** `PiCameraV3` (`picamera2`/`libcamera`), `config/raspberry_pi.yaml`, `tools/read_fc_yaw_params.py`, `src/web_server.py` (tắt mặc định trong `simulation.yaml`).
- **Chỉ dùng trên Laptop mô phỏng:** `GazeboCamera` (`gz.transport13`), `config/simulation.yaml`.

---

## 3. Kiểm thử Độc lập & Bộ Test Hồi quy

```bash
cd pbl6_tricopter_vision

# Self-test từng module
python3 src/config_loader.py
python3 src/ego_motion.py
python3 src/controller.py
python3 src/camera.py --test-mock
python3 src/mavlink_client.py --test-mock
python3 src/detector.py
python3 src/tracker.py
python3 src/visualizer.py
python3 src/web_server.py
python3 src/metrics.py

# Bộ test thuật toán & mô phỏng vòng kín (chạy được trên cả RPi 5 lẫn Laptop)
python3 tests/test_ego_motion.py
python3 tests/test_tracker_states.py
python3 tests/test_bytetrack_ego.py
python3 tests/sim_yaw_loop.py
```

---

## 4. Hướng dẫn Vận hành

### 4.1. Chạy thực tế trên Raspberry Pi 5 (Camera Module v3 + Pixhawk 6C)
```bash
# Tự chọn người thủ công trên Flask + ghi log đánh giá (khuyên dùng khi bay thử)
python3 main.py --profile raspberry_pi --no-auto-lock --debug

# Chạy mặc định (auto_lock: true theo raspberry_pi.yaml)
python3 main.py --profile raspberry_pi
```
* Mặc định kết nối `/dev/ttyAMA0 @ 921600`, `guided_vel`, Telemetry `ATTITUDE 50 Hz`, Camera v3 `640x480 @ 20fps` (`rotate_180=true`, cố định nét `af_mode: manual` `0.3D`, phơi sáng nhanh `ae_exposure_mode: short`), Web Stream `http://<IP_RPi>:5000`, HUD tối giản (`hud_mode: minimal`), tắt log định kỳ (`terminal_log_interval: 0.0`).
* **Lưu ý an toàn khi kết thúc phiên bay `--debug`:** Hạ cánh → Disarm → bấm **`Ctrl + C` đúng 1 lần và đợi in xong `=== BÁO CÁO DEBUG HỆ THỐNG ===`** rồi mới ngắt pin (tránh mất `report.txt` / `steps.csv`).
* **Kiểm tra tham số Yaw trên Pixhawk (chỉ đọc):**
  ```bash
  python3 tools/read_fc_yaw_params.py
  ```

### 4.2. Chạy Mô phỏng trên Laptop (Gazebo + ArduPilot SITL)
```bash
# Khuyên bật --hud-mode full và --debug để xem đầy đủ thông số & lưu báo cáo
python3 main.py --profile simulation --hud-mode full --debug
```
* *Lưu ý:* `config/common.yaml` hiện đang đặt `min_vx: 0`, `max_vx: 0` (khóa tiến/lùi để kiểm thử riêng trục Yaw). Nếu muốn mô phỏng cả tiến/lùi trên Gazebo, chỉ ghi đè `min_vx` / `max_vx` trong `config/simulation.yaml`.

### 4.3. Chạy thử không cần phần cứng (Mock Camera + Dry-run MAVLink)
```bash
python3 main.py --profile mock --gui --hud-mode full --dry-run-mavlink --debug
```

---

## 5. Debug, Đánh giá & Xuất Đồ thị (`--debug`)

- Khi chạy kèm `--debug`, kết quả lưu vào `logs/<timestamp>/` gồm `frames.csv`, `steps.csv` và `report.txt`.
- Vẽ 3 biểu đồ PNG (`01_overview.png`, `02_latency_distribution.png`, `03_tracking_control.png`):
  ```bash
  python3 plot_logs.py --log-dir logs/<timestamp>
  ```
  *(Yêu cầu cài thêm `matplotlib` trên môi trường chạy `plot_logs.py`.)*
- Mở trực tiếp `report_export_pack.html` bằng trình duyệt bất kỳ (offline) để lọc số liệu đo thực tế trên RPi 5 và tải đồ thị báo cáo.
