# Tricopter Person Detection & Tracking (Gazebo SITL & Raspberry Pi 5 + Pixhawk 6C)

Dự án hệ thống **nhận diện và bám theo mục tiêu (con người)** sử dụng **YOLOv8n + ByteTrack** và **ArduPilot**. Bộ mã nguồn được chia thành các module độc lập, vừa phục vụ kiểm thử trong **Gazebo SITL** vừa chạy trực tiếp trên **Raspberry Pi 5 (Camera Module v3) + Pixhawk 6C** cho dự án **Tricopter**.

---

## 1. Cấu trúc Module Hoá (9 Module Độc lập & Tự Kiểm thử)

Mỗi file trong `src/` đảm nhận một chức năng rõ ràng và đều tích hợp sẵn kịch bản tự kiểm thử (`python3 src/<tên_file>.py`):

```text
pbl6_tricopter_dat/
├── models/
│   └── yolov8n.pt              # Trọng số mô hình YOLOv8 Nano
├── src/
│   ├── controller.py           # [Module 1] Bộ điều khiển PID/PD cơ sở & cấu trúc ControlCommand
│   ├── selector.py             # [Module 2] Lựa chọn & khóa ID mục tiêu (TargetSelector)
│   ├── yaw_tracker.py          # [Module 3] Điều khiển ĐỘC LẬP chỉ riêng góc quay Yaw (deg/s)
│   ├── distance_tracker.py     # [Module 4] Điều khiển ĐỘC LẬP tiến/lùi Vx (Auto-Ref + Deadband ±12px)
│   ├── tracker.py              # [Module 5] Tracker CHUNG kết hợp Selector + Yaw + Distance cho Gazebo
│   ├── camera.py               # [Module 6] Trừu tượng hóa Camera (GazeboCamera & RPi5 PiCameraV3)
│   ├── mavlink_client.py       # [Module 7] Giao tiếp MAVLink (SITL UDP & Pixhawk 6C UART)
│   ├── detector.py             # [Module 8] Nhận diện & định danh người (YOLOv8n + ByteTrack)
│   ├── visualizer.py           # [Module 9] Vẽ HUD (Deadband Yaw & Tiến/Lùi, ID người, FPS) & log Terminal
│   └── main_tracker.py         # Chương trình điều phối chính (--mode sim | --mode real)
├── requirements.txt            # Các thư viện Python phụ thuộc
└── README.md                   # Tài liệu hướng dẫn và mô tả dự án
```

---

## 2. Cơ chế Bám Khoảng cách Tiến/Lùi Không Bị Tự Trôi (`src/distance_tracker.py`)

Để khắc phục hiện tượng drone tự trôi về một hướng hoặc không lùi lại khi người tiến gần:
1. **Tự động chốt chiều cao chuẩn (`Auto-Calibrate Reference Height`):**
   * Ngay tại thời điểm khóa mục tiêu (hoặc khi bấm phím `D` / `R`), hệ thống tự động lấy chiều cao khung bao hiện tại của người làm mốc chuẩn `ref_height_px` (ví dụ `130px`).
   * Tại vị trí đó sai số bằng `0px` $\rightarrow$ **`Vx = 0.0 m/s` (Drone đứng yên giữ cự ly, không bị tự trôi)**.
2. **Vùng chết Tiến/Lùi (`dist_deadband_px = ±12px`):**
   * Nếu chiều cao người dao động trong khoảng `[ref_height_px - 12px, ref_height_px + 12px]` $\rightarrow$ `Vx = 0.0 m/s` (`HOLD`).
   * Nếu người **bước lại gần** (`current_h > ref_height_px + 12px`) $\rightarrow$ **`Vx < 0` (Drone lùi lại ngay)**.
   * Nếu người **đi ra xa** (`current_h < ref_height_px - 12px`) $\rightarrow$ **`Vx > 0` (Drone tiến theo)**.
3. **Bù cắt mép dưới khung hình (`Bottom-Edge Clipping Compensation`):**
   * Khi người tiến sát làm chân chạm đáy ảnh (`y2 >= 472`), hệ thống tự động ước lượng chiều cao thực qua bề ngang khung bao (`width * aspect_ratio`), đảm bảo drone luôn nhận biết người đang ở rất gần và lùi lại.

---

## 3. Quy trình Kiểm thử Từng Module (Step-by-Step Verification)

```bash
python3 src/controller.py
python3 src/selector.py
python3 src/yaw_tracker.py
python3 src/distance_tracker.py
python3 src/tracker.py
python3 src/camera.py --test-mock
python3 src/mavlink_client.py --test-mock
python3 src/detector.py
python3 src/visualizer.py
```

---

## 4. Hướng dẫn Vận hành & Phím tắt Điều khiển

### 4.1. Chạy trong Mô phỏng Gazebo + ArduPilot SITL
```bash
# Chạy kèm cả xoay Yaw + Tiến/Lùi giữ khoảng cách:
python3 src/main_tracker.py --mode sim --distance

# Hoặc chạy chỉ xoay Yaw trước (nhấn phím 'D' trên màn hình để bật Tiến/Lùi bất kỳ lúc nào):
python3 src/main_tracker.py --mode sim
```

### 4.2. Chạy trên Thực tế: Raspberry Pi 5 (Camera v3) + Pixhawk 6C
```bash
python3 src/main_tracker.py --mode real --mavlink /dev/ttyAMA0 --baud 921600 --imgsz 320
```

### 4.3. Bảng Phím tắt & Tương tác Chuột trên Cửa sổ Camera

| Thao tác | Chức năng |
| :--- | :--- |
| **Click Chuột Trái** vào người | Khóa (`LOCKED`) mục tiêu và chốt cự ly hiện tại làm chuẩn |
| **Click Chuột Phải** (hoặc phím `C` / `Space`) | Hủy khóa ngay lập tức $\rightarrow$ Drone **Hover** chờ chọn lại |
| Phím **`Tab`** hoặc **`N`** | Chuyển đổi luân phiên sang mục tiêu (`track_id`) kế tiếp |
| Phím **`A`** | Tự động chấm điểm và khóa mục tiêu tối ưu nhất (Gần tâm + Gần drone) |
| Phím **`D`** | **Bật / Tắt nhanh** chế độ điều khiển tiến/lùi (`Vx`) và chốt cự ly hiện tại |
| Phím **`R`** | **Chốt lại (Recalibrate)** khoảng cách hiện tại của người làm mốc chuẩn mới |
| Phím **`+` (`=`)** / **`-`** | Chỉnh mốc chuẩn để giữ khoảng cách **Gần hơn** (`+10px`) / **Xa hơn** (`-10px`) |
| Phím **`Q`** | Dừng drone (Hover) và thoát chương trình |
