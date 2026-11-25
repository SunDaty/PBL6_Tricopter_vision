# AGENT SYNC: Giao tiếp giữa Agent Raspberry Pi 5 (Thực tế) & Agent Laptop (Mô phỏng)

> **Mục đích:** Tệp điều phối chính nằm trong GitHub repository để **Agent RPi 5** (bay thực tế trên Tricopter) và **Agent Laptop** (chạy mô phỏng Gazebo + ArduPilot SITL) trao đổi ngữ cảnh, ràng buộc phần cứng, kết quả kiểm thử và danh sách sửa lỗi mà không làm hỏng môi trường của nhau.
>
> **Quy trình bắt buộc cho cả 2 Agent:**
> 1. Sau khi `git pull`, đọc tệp này trước khi sửa bất kỳ mã nguồn nào.
> 2. Trước khi `git push`, thêm một mục mới vào **Mục 6 (Nhật ký bàn giao / Handoff Log)** ghi rõ commit, kết quả chạy/đo và lưu ý cho Agent phía bên kia.

---

## 1. Phân vai & Môi trường của 2 Agent

| Thuộc tính | Agent RPi 5 (Thực tế) | Agent Laptop (Mô phỏng) |
|---|---|---|
| **Vai trò** | Triển khai bay thực tế, tối ưu tài nguyên nhúng, đo log phần cứng thật | Chạy mô phỏng Gazebo + ArduPilot SITL, thử nghiệm thuật toán điều khiển/tracking mới, đo đáp ứng vòng kín (`E2E`, `Rise`, `Settling`, `Overshoot`) |
| **Profile cấu hình** | `--profile raspberry_pi` (`config/raspberry_pi.yaml`) | `--profile simulation` (`config/simulation.yaml`) hoặc `--profile mock` |
| **Nguồn ảnh** | `PiCameraV3` (`picamera2` / CSI, 640×480 @ 20 FPS) | `GazeboCamera` (`gz.transport13`, topic `/camera/image_raw`) hoặc `MockCamera` |
| **MAVLink** | Pixhawk 6C UART `/dev/ttyAMA0 @ 921600` | ArduPilot SITL UDP `udpin:0.0.0.0:14550` |
| **Hiển thị** | Headless + Flask MJPEG (`port 5000`, `hud_mode: minimal`) | Cửa sổ OpenCV GUI (`headless: false`, khuyên dùng `--hud-mode full`) |

---

## 2. Phân định Module: Cần mô phỏng vs Chỉ dùng cho phần cứng RPi 5

Mọi mã nguồn đều đã đẩy đầy đủ lên GitHub. Nhờ cơ chế **lazy import** (`picamera2` chỉ import trong `PiCameraV3.start()`, `gz.transport13` chỉ import trong `GazeboCamera.start()`), Laptop có thể `git pull` toàn bộ repo mà **không cần cài thư viện của Raspberry Pi**.

| Module / Tệp | Dùng cho Mô phỏng (Laptop) | Dùng cho Thực tế (RPi 5) | Ghi chú |
|---|:---:|:---:|---|
| `main.py`, `src/main_tracker.py` | **Có** | **Có** | Điều phối chính dùng chung 100% |
| `src/config_loader.py`, `config/common.yaml` | **Có** | **Có** | Cấu hình nền dùng chung |
| `src/detector.py` + `models/yolov8n_ncnn_model/` | **Có** | **Có** | YOLOv8n NCNN + ByteTrack bù ego-motion |
| `src/ego_motion.py` | **Có** | **Có** | `YawHistory` + `BearingProjector` |
| `src/tracker.py`, `src/controller.py` | **Có** | **Có** | Máy trạng thái `TRACK/COAST/HOLD`, `RE-ACQ`, bù trễ, Hysteresis, Slew |
| `src/mavlink_client.py` | **Có** | **Có** | Giao tiếp `SET_POSITION_TARGET_LOCAL_NED` + đọc `ATTITUDE`/`TIMESYNC` |
| `src/visualizer.py`, `src/metrics.py`, `plot_logs.py` | **Có** | **Có** | Vẽ HUD, đo CSV/report, xuất đồ thị (`plot_logs.py` cần `matplotlib`) |
| `tests/*.py` | **Có** | **Có** | 4 bài test hồi quy thuật toán (chạy độc lập không cần phần cứng) |
| `config/simulation.yaml`, `GazeboCamera` (`src/camera.py`) | **Có** | Không | Chỉ dùng trên Laptop |
| `config/raspberry_pi.yaml`, `PiCameraV3` (`src/camera.py`) | **Không cần** | **Có** | Không cần cài `picamera2` / `libcamera` trên Laptop |
| `src/web_server.py` | Tùy chọn | **Có** | Tắt mặc định trong `simulation.yaml` (`web.enabled: false`); Laptop dùng OpenCV GUI |
| `tools/read_fc_yaw_params.py` | Không cần *(trừ khi truyền `--conn udpin:...`)* | **Có** | Mặc định đọc tham số Pixhawk qua `/dev/ttyAMA0` |
| `rpi5_measured_summary.md`, `report_best_run_*.md`, `report_export_pack.html` | Tham khảo báo cáo | Xuất từ RPi 5 | Số liệu đo thực tế để đối chiếu cột Thực tế vs Mô phỏng |

---

## 3. Ràng buộc Cứng (Hard Constraints — Cả 2 Agent PHẢI tuân thủ)

1. **KHÔNG xóa cấu hình chống sụt áp RPi 5 (Anti-Brownout):**
   - Giữ `cv2.setNumThreads(1)` trong `src/detector.py`, `ncnn_threads: 2`, `lightmode = True`, `fp16 = True` mặc định trong `common.yaml`.
   - Giữ cơ chế chỉ nén JPEG khi `active_clients > 0` trong `src/web_server.py`.
   - Nếu Agent Laptop muốn chạy nhanh hơn trên máy mạnh (ví dụ `ncnn_threads: 4`, `target_fps: 30`, `hud_mode: full`), hãy **chỉ ghi đè trong `config/simulation.yaml`**, không làm tăng tải mặc định cho `raspberry_pi.yaml`.
2. **Giữ Lazy Import cho thư viện đặc thù từng máy:**
   - Tuyệt đối không đưa `import picamera2` hay `from gz.transport13 import Node` lên đầu file `src/camera.py`.
3. **Giữ tương thích ngược các khóa cấu hình YAML cũ:**
   - Giữ nguyên tên khóa `yaw_kp`, `yaw_kd`, `yaw_deadband_px`, `max_yaw_rate`, `min_vx`, `max_vx` trong `config/common.yaml`. Tham số mới phải có giá trị mặc định an toàn.
4. **An toàn khi mở khóa `min_vx` / `max_vx`:**
   - Hiện `config/common.yaml` đang đặt **`min_vx: 0`, `max_vx: 0`** để khóa trục tiến/lùi trên drone thật.
   - Nếu Agent Laptop muốn test tiến/lùi trong Gazebo, hãy đặt `tracker.min_vx` / `tracker.max_vx` **bên trong `config/simulation.yaml`** để tránh việc RPi 5 `git pull` về rồi vô tình bay lao tới khi người dùng tưởng vẫn đang khóa Yaw!
5. **Kiểm thử bắt buộc trước khi push:**
   - Chạy pass toàn bộ `tests/test_ego_motion.py`, `tests/test_tracker_states.py`, `tests/test_bytetrack_ego.py`, `tests/sim_yaw_loop.py` và các self-test trong `src/`.

---

## 4. Hướng dẫn nhanh cho Agent Laptop chạy thử ngay phiên bản hiện tại

```bash
# 1. Kéo code mới nhất (lưu ý cài thêm matplotlib nếu muốn chạy plot_logs.py)
git pull origin main

# 2. Chạy bộ kiểm thử đơn vị & mô phỏng vòng kín (không cần Gazebo)
python3 tests/test_ego_motion.py
python3 tests/test_tracker_states.py
python3 tests/test_bytetrack_ego.py
python3 tests/sim_yaw_loop.py

# 3. Chạy mô phỏng Gazebo + ArduPilot SITL (khuyên thêm --hud-mode full --debug)
python3 main.py --profile simulation --hud-mode full --debug

# 4. Hoặc chạy nhanh không cần Gazebo/SITL
python3 main.py --profile mock --gui --hud-mode full --dry-run-mavlink --debug
```

**Lưu ý khi chạy `--profile simulation` hiện tại:**
- `common.yaml` đang đặt `min_vx: 0`, `max_vx: 0` → drone trong Gazebo sẽ **chỉ quay Yaw** bám người.
- `common.yaml` đang đặt `hud_mode: minimal` và `terminal_log_interval: 0.0` → thêm `--hud-mode full` nếu muốn xem bảng thông số trực tiếp trên cửa sổ OpenCV.
- Khi kết thúc đo trong Gazebo, nhấn **`q` trên cửa sổ OpenCV** hoặc **`Ctrl + C` 1 lần và đợi in xong báo cáo** để xuất đủ `frames.csv`, `steps.csv`, `report.txt`.

---

## 5. Danh sách việc cần phối hợp (Backlog chung)

### Nhiệm vụ ưu tiên gửi cho Agent Laptop (test trên Gazebo SITL)
1. **Đo bộ số liệu chuẩn cột "Mô phỏng (Gazebo + SITL)" cho báo cáo:**
   - Chạy `--profile simulation --debug`, hover ổn định ở `GUIDED`, click chọn người và thực hiện **4–6 bước nhảy Yaw (Step Test)** (`|err_x|` nhảy từ `<15 px` sang `120–180 px`, đứng yên 5–6 s cho drone xác lập) để thu đủ `Infer FPS`, `Pipeline FPS`, `Capture->cmd`, `Cmd->response`, `E2E`, `Rise time`, `Settling time`, `Overshoot (%)`, `Yaw overshoot (°)`.
2. **Kiểm chứng & sửa 4 bug logic đã phát hiện trong quá trình review trên RPi 5** (ưu tiên test kỹ trong Gazebo trước khi RPi 5 kéo về bay thật):
   - **`BUG-1` (`src/main_tracker.py:272, 336`):** `self._latest_frame = frame` lưu tham chiếu rồi `draw_hud(frame, ...)` vẽ đè in-place ngoài `_tracker_lock` → race condition & làm bẩn histogram màu áo khi click chọn trên Flask/OpenCV. (Sửa: copy ROI hoặc lưu bản sao chỉ khi `need_draw`, hoặc vẽ lên `display_frame = frame.copy()` khi cần, giữ `self._latest_frame` sạch).
   - **`BUG-2` (`src/tracker.py:521`):** `self._coexisting_ids` bị ghi đè sạch mỗi frame (không có TTL) và `_try_reacquire` chạy ngay ở frame mất đầu tiên (không chờ grace period 2–3 frame `COAST`) → đã gây nhảy ping-pong `#12 ↔ #13` trong log thực tế `20261119_143256`.
   - **`BUG-3` (`src/tracker.py:535-549`):** Khi `psi_capture` chuyển từ `None` sang có góc thật (như 42 khung đầu ở `20261119_143634`), `_theta_target` chưa được tái neo (re-anchor) gây giật `_omega_person`.
   - **`BUG-4` (`src/tracker.py:475`):** Khi vào `COAST`/`SEARCH_HOLD`, `self._current_vx = 0.0` phanh bậc thang bỏ qua `max_vx_decel` (cần sửa sang slew-rate + bổ sung Yaw-Priority Scaling trước khi mở khóa `vx` bay thật).
   - **`SAFETY-1` (`src/mavlink_client.py:90, 227-234`):** Bổ sung đọc `HEARTBEAT` để nhận biết mode `GUIDED`/`LOITER` + `ARMED` (reset `yaw_slew` khi vừa chuyển sang `GUIDED`), nhả `rc_override` (`65535`) khi `IDLE`, và chặn gửi `guided_nogps` nếu `nogps_thrust <= 0.0`.
3. **Cập nhật `requirements.txt`:** bổ sung `matplotlib` (cho `plot_logs.py`).

### Nhiệm vụ thuộc Agent RPi 5 (phần cứng thật)
1. Tách luồng grabber camera cho `PiCameraV3` (cắt ~37 ms trễ hàng đợi buffer) + chuyển xoay 180° sang `libcamera.Transform(hflip=1, vflip=1)` + watchdog timeout camera.
2. Chạy `tools/read_fc_yaw_params.py` trên Pixhawk 6C thật để kiểm tra nguyên nhân dao động nền ~0.8 Hz ở vòng Yaw.
3. Bay thật 1 phiên Step Test sạch (kết thúc bằng `Ctrl+C` trước khi rút pin) để lấy `E2E` và `Settling time` bay thật.

---

## 6. Nhật ký Bàn giao (Handoff Log)

### [2026-11-25 22:40 +07] — Từ: **Agent RPi 5 (Thực tế)** → Tới: **Agent Laptop (Mô phỏng)**
- **Commit nền:** `5af5310` (code + tests + tools + `plot_logs.py`) và `2baac94` (bộ số liệu đo thực tế RPi 5).
- **Tóm tắt thay đổi vừa đẩy:**
  - Thêm `src/ego_motion.py` (`YawHistory` 50 Hz + `BearingProjector`), bù ego-motion cho ByteTrack (`detector.py`, hạ `conf_thresh: 0.20`, `act: 0.45`), bù độ trễ + `HysteresisDeadband` + `SlewRateLimiter` + máy trạng thái `TRACK/COAST/SEARCH_HOLD` + `RE-ACQ` (`tracker.py`, `controller.py`).
  - Cố định camera trên RPi 5 (`af_mode: manual 0.3D`, `ae_exposure_mode: short`), HUD `minimal` không `frame.copy()`/`addWeighted`, tắt log định kỳ (`terminal_log_interval: 0.0`).
  - Toàn bộ 13/13 unit & simulation test đều PASS trên RPi 5.
- **Số liệu thực tế chuẩn đã đo trên RPi 5 (`logs/20261119_143256`, lưu trong `report_export_pack.html`):**
  - `Infer FPS = 11.7` (`infer_ms`: mean `85.8`, P50 `85.7`, P95 `87.6 ms`), `Pipeline FPS = 10.9` (CSV `11.26`).
  - `Capture->bbox = 123.4 ms` (P95 `145.6`), `Capture->cmd = 123.6 ms` (P50 `123.4` / P95 `145.8 ms`; overhead tính điều khiển + MAVLink `0.21 ms`).
  - `TIMESYNC OK` (`RTT min = 4.32 ms`), mất track `27.98%` (`TRACK 72.0% | COAST 8.0% | HOLD 19.9%`), `lock_loss = 12`, **nghi đổi ID = `0`**, **`RE-ACQ = 5`**, nguồn `throttled=0x0`.
  - *Lưu ý:* Phiên `143256` là test mặt đất/xoay tay nên chưa có `Settling time` bay thật; 2 phiên bay thật ngày 17/11 (`212706`, `132532`) chạy bản code cũ và thiếu `report.txt` do ngắt pin đột ngột.
- **Yêu cầu cho Agent Laptop:**
  - Chạy thử mô phỏng code hiện tại trên Gazebo + ArduPilot SITL, ghi lại bảng số liệu mô phỏng vào mục tiếp theo bên dưới.
  - Nếu có sửa các lỗi `BUG-1..4` / `SAFETY-1` hoặc thử mở `vx` trong `simulation.yaml`, nhớ giữ nguyên các Hard Constraints ở Mục 3 và chạy đủ bộ `tests/*.py` trước khi push.

---

### [TEMPLATE — Agent Laptop điền sau khi chạy mô phỏng / chỉnh sửa]
- **Thời gian & Commit:** `<YYYY-MM-DD HH:MM> | <commit_hash>`
- **Kịch bản mô phỏng đã chạy:** `<Gazebo world, ArduPilot SITL version, profile, CLI flags>`
- **Kết quả đo mô phỏng (`report.txt` / `steps.csv`):**
  - `Infer FPS / Pipeline FPS`: `...`
  - `Capture->cmd (P50 / P95)`: `...`
  - `Cmd->response / E2E (n, mean / P95)`: `...`
  - `Mất track (%) / lock_loss / Nghi đổi ID / RE-ACQ`: `...`
  - `Yaw Step (số bước / settled / Rise / Settling / Overshoot% / Yaw OS°)`: `...`
- **Các thay đổi code / config (nếu có):** `...`
- **Lưu ý / Cảnh báo gửi lại cho Agent RPi 5 trước khi kéo về bay thật:** `...`
