# Tổng hợp Thông số Đo thực tế — `pbl6_tricopter_vision` (Raspberry Pi 5 + Pixhawk 6C)

> Tài liệu chỉ tổng hợp số liệu thực tế đọc trực tiếp từ `logs/`, `config/` và phần cứng Raspberry Pi 5 (không sửa mã nguồn, không nội suy/bịa số liệu).  
> Phạm vi quét: toàn bộ `/home/dat/drone_tracking` — tất cả **11 thư mục log** đều nằm trong `pbl6_tricopter_vision/logs/`; **không tồn tại** thư mục `src/debug_logs/` hay file log ở vị trí khác.

---

## Bảng 1: Danh sách và Phân loại các Phiên chạy trong `logs/`

| # | Thư mục `logs/<timestamp>` | Tệp hiện có | Thời lượng (s) | Tổng số khung | `yaw_att_deg` (có/tổng) | Khung khóa `locked_id` | `TIMESYNC` (RTT min) | Mẫu `E2E` (`n` / timeout) | Bước yaw (`steps.csv` / `settled`) | Phân loại & Lý do chọn / loại |
|---|---|---|---:|---:|---:|---:|---|---|---:|---|
| 1 | `20261117_114848` | `frames.csv`, `report.txt`, `steps.csv` | 108.6 *(CSV: 105.2)* | 873 | 873 / 873 | 801 | OK (1.82 ms) | `n=2` / 19 | 4 / 0 | **CHỌN (Đợt 1 — Trước tối ưu, đủ 3 tệp)**. Có khóa mục tiêu, đồng bộ đồng hồ, `E2E` và `steps.csv`. Lưu ý: `\|yawspeed\|` P95 = 1.14 °/s (test bàn/tĩnh, chưa bay quay mạnh; lúc này chưa khóa `vx`, `max\|vx\| = 1.20 m/s`). |
| 2 | `20261117_132203` | `frames.csv` (0 B) | 0 | 0 | 0 / 0 | 0 | Chưa có | Chưa có | Chưa có | **LOẠI** — Tệp rỗng, dừng/lỗi ngay khi khởi chạy. |
| 3 | `20261117_132532` | `frames.csv` | 206.3 *(CSV)* | 1,759 | 1,759 / 1,759 | 77 | Chưa có *(thiếu `report.txt`)* | Chưa có | Chưa có | **CHỌN CÓ ĐIỀU KIỆN (Đợt 1 — Bay thật)**. Có động học bay thật (`\|yawspeed\|` P95 = 40.9 °/s, max 137.5 °/s), có khóa mục tiêu (`77` khung), nhưng thiếu `report.txt`/`steps.csv` do thoát đột ngột. |
| 4 | `20261117_211409` | `frames.csv`, `report.txt`, `steps.csv`, `plots/` | 446.5 *(CSV: 443.3)* | 4,016 | 4,016 / 4,016 | **0** | OK (1.75 ms) | `n=0` / 0 | 0 / 0 | **LOẠI khỏi thống kê bám mục tiêu** — Chạy dài có thái độ từ Pixhawk nhưng không khóa mục tiêu (`locked_id` rỗng toàn bộ 4,016 khung, `yaw_cmd = 0`). Chỉ dùng tham khảo FPS/độ trễ suy luận. |
| 5 | `20261117_212706` | `frames.csv`, `plots/` | 342.9 *(CSV)* | 3,289 | 3,289 / 3,289 | 1,194 | Chưa có *(thiếu `report.txt`)* | Chưa có | Chưa có | **CHỌN (Đợt 1 — Bay thật dài nhất có khóa mục tiêu)**. `\|yawspeed\|` P95 = 26.7 °/s, max 53.3 °/s; 397 khung phát lệnh `yaw_cmd ≠ 0`, `vx = 0`. Thiếu `report.txt`/`steps.csv` do thoát đột ngột. |
| 6 | `20261119_143115` | `frames.csv` (header) | 0 | 0 | 0 / 0 | 0 | Chưa có | Chưa có | Chưa có | **LOẠI** — Chỉ có dòng tiêu đề CSV, 0 khung dữ liệu. |
| 7 | `20261119_143256` | `frames.csv`, `report.txt`, `steps.csv` | 84.3 *(CSV: 81.8)* | 922 | 922 / 922 | 672 | OK (4.32 ms) | `n=4` / 3 | 4 / 0 | **CHỌN (Đợt 2 — Sau tối ưu, phiên chuẩn đầy đủ nhất)**. Đủ 3 tệp, có khóa mục tiêu, đồng bộ đồng hồ, `E2E n=4`, `RE-ACQ = 5`, có đoạn xoay nhanh (`\|yawspeed\|` max 51.8 °/s, P95 = 5.86 °/s). |
| 8 | `20261119_143634` | `frames.csv`, `report.txt`, `steps.csv` | 30.3 *(CSV: 16.5)* | 188 | 146 / 188 | 188 | OK (4.74 ms) | `n=0` / 1 | 2 / 0 | **CHỌN CÓ ĐIỀU KIỆN (Đợt 2 — Test ngắn tĩnh)**. 42 khung đầu chưa nhận `yaw_att_deg`; `\|yawspeed\|` max 1.15 °/s (đặt tĩnh), `Pipeline FPS` trong `report.txt` bị thấp (6.2) do tính cả ~13.8 s khởi tạo (tốc độ thật từ `t_capture` là 11.37 FPS). |
| 9 | `20261119_143708` | `frames.csv`, `report.txt`, `steps.csv` | 34.7 *(CSV: 33.0)* | 374 | 374 / 374 | 367 | OK (4.89 ms) | `n=0` / 2 | 2 / 0 | **CHỌN (Đợt 2 — Sau tối ưu, test tĩnh/bàn)**. Đủ 3 tệp, `TIMESYNC OK`, khóa 367/374 khung; `\|yawspeed\|` max 2.82 °/s nên chưa có phản hồi drone (`E2E n=0`). |
| 10 | `20261119_143916` | `frames.csv`, `report.txt`, `steps.csv` | 24.3 *(CSV: 21.9)* | 249 | 249 / 249 | 249 | OK (4.67 ms) | `n=0` / 3 | 2 / 0 | **CHỌN (Đợt 2 — Sau tối ưu, test tĩnh/bàn ngắn)**. Đủ 3 tệp, `TIMESYNC OK`, khóa 249/249 khung, `RE-ACQ = 1`; `\|yawspeed\|` max 1.18 °/s nên `E2E n=0`. |
| 11 | `20261119_234333` | `frames.csv` (0 B) | 0 | 0 | 0 / 0 | 0 | Chưa có | Chưa có | Chưa có | **LOẠI** — Lỗi không phát hiện camera (`Picamera2` `IndexError`) lúc khởi động. |

> **Ghi chú phân đợt quan trọng cho báo cáo:**
> - **Đợt 1 (17/11/2026)** chạy trên cấu hình gốc (`HEAD` Git: `conf_thresh=0.45`, `yaw_kd=0.01`, `max_yaw_rate=25.0 °/s`, `lost_timeout=2.0 s`, `AfMode=2`, full HUD, không bù ego-motion/trễ). Trong đó `20261117_212706` và `20261117_132532` là **bay thật** nhưng thoát đột ngột nên thiếu `report.txt`/`steps.csv`.
> - **Đợt 2 (19/11/2026)** chạy trên cấu hình hiện tại sau tối ưu (`conf_thresh=0.20`, `ego_motion_comp=true`, `latency_comp=true`, `COAST/RE-ACQ`, `AfMode=manual`, `AE=short`, `minimal HUD`, tắt log định kỳ). Các phiên ngày 19/11 đều có đủ 3 tệp nhưng chủ yếu là **test bàn / xoay tay** (`|yawspeed|` nhỏ hoặc chỉ xoay ngắn), chưa có phiên bay hover ngoài trời mới.

---

## Bảng 2: Tổng hợp Các Chỉ số Đo thực tế (khớp từng dòng điền vào báo cáo)

### 2A. Bảng đại diện tổng hợp (dùng trực tiếp điền cột *"Thực tế (RPi 5 + Pixhawk 6C)"*)

*(Tách rõ **Đợt 1 — Trước tối ưu** và **Đợt 2 — Sau tối ưu** để báo cáo đối chứng chính xác và trung thực; các chỉ số trễ bỏ khung warm-up `frame 0` đúng theo logic `metrics.py`.)*

| Chỉ số đánh giá | Đợt 1 — Bay thật tiêu biểu (`logs/20261117_212706`, từ `frames.csv`) | Đợt 1 — Phiên đủ `report.txt` (`logs/20261117_114848`) | Đợt 2 — Phiên chuẩn đầy đủ nhất (`logs/20261119_143256`) | Đợt 2 — Gộp 4 phiên hợp lệ ngày 19/11 (`143256`+`143634`+`143708`+`143916`) |
|---|---|---|---|---|
| **Nguồn dữ liệu** | `frames.csv` (3,289 khung; 342.9 s) | `report.txt` + `frames.csv` + `steps.csv` (873 khung; 108.6 s) | `report.txt` + `frames.csv` + `steps.csv` (922 khung; 84.3 s) | Gộp `frames.csv` 4 phiên (1,733 khung; 153.2 s) |
| **Infer FPS (trung bình, `1000/mean`)** | **10.32 FPS** | **9.0 FPS** *(chính xác: 8.98)* | **11.7 FPS** *(chính xác: 11.65)* | **11.64 FPS** |
| **Thời gian suy luận `infer_ms` (mean / P50 / P95 / max)** | **96.9 / 89.9 / 130.1 / 265.5 ms** | **111.4 / 110.6 / 141.7 / 155.5 ms** | **85.8 / 85.7 / 87.6 / 103.3 ms** | **85.9 / 85.7 / 88.0 / 103.3 ms** |
| **Loop FPS / Pipeline FPS (trung bình)** | **9.59 FPS** *(từ `t_capture`)* | **8.0 FPS** *(report.txt)* / **8.29 FPS** *(CSV)* | **10.9 FPS** *(report.txt)* / **11.26 FPS** *(CSV)* | **11.31 FPS** *(từ `t_capture`)* |
| **Độ trễ `Capture->bbox` (mean / P50 / P95 / max)** | **136.0 / 136.7 / 172.8 / 308.7 ms** | **149.9 / 148.6 / 190.2 / 217.7 ms** | **123.4 / 123.2 / 145.6 / 157.3 ms** | **123.5 / 123.2 / 145.5 / 157.3 ms** |
| **Độ trễ `Capture->cmd sent` (P50 / P95)** *(kèm mean)* | **136.9 / 173.0 ms** *(mean 136.2)* | **148.8 / 190.3 ms** *(mean 150.1)* | **123.4 / 145.8 ms** *(mean 123.6)* | **123.4 / 145.7 ms** *(mean 123.7)* |
| **`TIMESYNC RTT min`** | Chưa có dữ liệu *(thiếu `report.txt`)* | **1.82 ms** *(OK)* | **4.32 ms** *(OK)* | **4.32 – 4.89 ms** *(OK cả 4 phiên)* |
| **Độ trễ `Cmd->drone response` (`n`, mean / P50 / P95 / max)** | Chưa có dữ liệu *(thiếu `report.txt`)* | `n=2` *(19 timeout)*: **316.6 / 316.6 / 347.3 / 350.7 ms** | `n=4` *(3 timeout)*: **640.7 / 651.8 / 1,160.3 / 1,164.9 ms** | Chỉ có ở `143256` (`n=4`); 3 phiên còn lại `n=0` |
| **Độ trễ `E2E capture->response` (`n`, mean / P50 / P95 / max)** | Chưa có dữ liệu *(thiếu `report.txt`)* | `n=2`: **463.7 / 463.7 / 497.6 / 501.3 ms** | `n=4`: **778.9 / 789.0 / 1,298.2 / 1,303.3 ms** | Chỉ có ở `143256` (`n=4`); 3 phiên còn lại `n=0` |
| **Tổng khung đang khóa (`locked_id ≠ ∅`)** | **1,194** / 3,289 khung | **801** / 873 khung | **672** / 922 khung | **1,476** / 1,733 khung |
| **Khung mất dấu tạm thời (`err_x = ∅`) & Tỉ lệ mất track (%)** | **770 khung (64.49%)** *(439 khung vẫn có `n_det > 0`)* | **293 khung (36.58%)** *(55 khung vẫn có `n_det > 0`)* | **188 khung (27.98%)** *(COAST 8.0%, HOLD 19.9%)* | **316 khung (21.41%)** |
| **Số lần mất khóa (`lock_loss`)** | **63** *(tính từ CSV)* | **49** *(report.txt & CSV khớp)* | **12** *(report.txt & CSV khớp)*; **RE-ACQ = 5** | **19** *(tổng)*; **RE-ACQ = 6** |
| **Số lần nghi đổi ID (`suspected_switches` / `id_switch_flag`) & Tỉ lệ (%)** | **10 lần** — **0.84%** *(trên khung khóa)* / **15.87%** *(trên số lần mất khóa)* | **5 lần** — **0.62%** *(trên khung khóa)* / **10.20%** *(trên số lần mất khóa)* | **0 lần (0.00%)** | **0 lần (0.00%)** |
| **Bước nhảy Yaw (`steps.csv`): Số bước / `settled`** | Chưa có dữ liệu *(thiếu `steps.csv`)* | **4 bước / 0 xác lập** | **4 bước / 0 xác lập** | **10 bước / 0 xác lập** |
| **Yaw `Rise time` (s)** | Chưa có dữ liệu | **0.52 s** *(1/4 bước đạt ngưỡng; 3 bước `nan`)* | **2.24 s** *(2/4 bước đạt: 4.39 s & 0.09 s; 2 bước `nan`)* | Chỉ 2/10 bước có giá trị (đều ở `143256`) |
| **Yaw `Settling time` (s)** | Chưa có dữ liệu | **Chưa có dữ liệu (`nan`)** | **Chưa có dữ liệu (`nan`)** | **Chưa có dữ liệu (`nan`)** |
| **Yaw `Overshoot` (%)** | Chưa có dữ liệu | **TB 18.3% (max 73.2%)** | **TB 22.7% (max 90.7%)** | 10 bước: 9 bước `0.0%`, 1 bước `90.7%` |
| **Yaw `Yaw overshoot` (°)** | Chưa có dữ liệu | **TB 0.0°** *(max 0.04°)* | **TB 0.0°** *(max 0.003°)* | **0.0° – 0.1°** *(max 0.06°)* |

---

### 2B. Chi tiết từng phiên hợp lệ còn lại (để tra cứu đối chiếu)

| Phiên `logs/<timestamp>` | Khung / Thời lượng | Infer FPS / `infer_ms` (mean/P50/P95/max) | Pipe FPS (rep / CSV) | `Cap->bbox` (mean/P50/P95/max) | `Cap->cmd` (P50/P95) | `TIMESYNC` / `E2E` | Khóa / Mất track (%) / `lock_loss` / Đổi ID / `RE-ACQ` | `steps.csv` (bước / settled / Rise / Settle / OS% / Yaw OS°) |
|---|---|---|---|---|---|---|---|---|
| `20261117_132532` *(Đợt 1, bay thật)* | 1,759 / 206.3 s | 9.02 / 110.9 / 102.2 / 139.1 / 304.1 ms | — / 8.52 | 148.9 / 151.1 / 185.2 / 347.5 ms | 151.2 / 185.4 ms | Chưa có / Chưa có | 77 / 59 (**76.62%**) / 5 / 1 (1.30% khung khóa; 20.0% mất khóa) / 0 | Chưa có dữ liệu *(thiếu `steps.csv`)* |
| `20261119_143634` *(Đợt 2, tĩnh ngắn)* | 188 / 30.3 s *(CSV 16.5 s)* | 11.5 / 86.8 / 86.4 / 91.0 / 101.2 ms | 6.2 / 11.37 | 125.8 / 125.0 / 146.3 / 149.6 ms | 125.2 / 146.5 ms | OK (4.74 ms) / `n=0` (1 to) | 188 / 0 (**0.00%**) / 0 / 0 (0%) / 0 | 2 / 0 / `nan` / `nan` / 0.0% / 0.1° |
| `20261119_143708` *(Đợt 2, tĩnh)* | 374 / 34.7 s *(CSV 33.0 s)* | 11.7 / 85.8 / 85.7 / 87.5 / 101.2 ms | 10.8 / 11.29 | 123.1 / 123.4 / 145.3 / 152.9 ms | 123.6 / 145.5 ms | OK (4.89 ms) / `n=0` (2 to) | 367 / 82 (**22.34%**) / 5 / 0 (0%) / 0 | 2 / 0 / `nan` / `nan` / 0.0% / 0.0° |
| `20261119_143916` *(Đợt 2, tĩnh ngắn)* | 249 / 24.3 s *(CSV 21.9 s)* | 11.7 / 85.6 / 85.5 / 88.4 / 99.0 ms | 10.2 / 11.33 | 122.8 / 122.3 / 145.0 / 147.2 ms | 122.5 / 145.1 ms | OK (4.67 ms) / `n=0` (3 to) | 249 / 46 (**18.47%**) / 2 / 0 (0%) / 1 | 2 / 0 / `nan` / `nan` / 0.0% / 0.0° |
| *(Tham khảo không khóa)* `20261117_211409` | 4,016 / 446.5 s | 9.7 / 103.1 / 95.4 / 133.4 / 217.7 ms | 9.0 / 9.06 | 143.4 / 145.2 / 180.5 / 269.4 ms | 145.4 / 180.7 ms | OK (1.75 ms) / `n=0` (0 to) | 0 / — / 0 / 0 / 0 | 0 bước *(không khóa mục tiêu)* |

### 2C. Giải thích nguyên nhân trực tiếp từ dữ liệu log đối với các chỉ số `Chưa có dữ liệu` / `nan` / `n=0`

1. **Thiếu `report.txt`, `steps.csv`, `TIMESYNC`, `E2E` ở hai phiên bay thật ngày 17/11 (`20261117_132532` và `20261117_212706`):**
   - Chương trình dừng đột ngột (ngắt nguồn / bị kill) nên không chạy khối `finally: metrics.close()` trong [main_tracker.py](file:///home/dat/drone_tracking/pbl6_tricopter_vision/src/main_tracker.py). Dữ liệu trong `frames.csv` vẫn ghi đầy đủ (0 dòng hỏng) nhưng `CmdResponseTracker`, `ClockSync` và `YawStepAnalyzer` chỉ xuất ra `report.txt` / `steps.csv` lúc đóng phiên.
2. **`Cmd->drone response` và `E2E` có `n=0` ở `20261119_143634`, `143708`, `143916` (và chỉ có `n=2` ở `114848`, `n=4` ở `143256`):**
   - Trong [metrics.py:185-212](file:///home/dat/drone_tracking/pbl6_tricopter_vision/src/metrics.py#L185-L212), `CmdResponseTracker` yêu cầu lệnh nhảy từ `< 1.0 °/s` lên `≥ 5.0 °/s` và tốc độ quay thực tế `yawspeed_att` cùng dấu phải đạt `≥ max(2.0, 0.2 × |cmd|)` trong vòng 2 s.
   - Ở `143634`, `143708`, `143916`, `|yawspeed_att|` cực đại chỉ đạt `1.15 – 2.82 °/s` (drone đặt tĩnh trên bàn / không xoay theo lệnh dù `yaw_cmd` lên tới `13.45 – 19.05 °/s`), dẫn tới toàn bộ bước lệnh bị hết thời gian chờ (`timeout = 1..3`) và `n = 0`.
   - Ở `114848` (test bàn, `|yawspeed|` P95 = `1.14 °/s`, 19 timeout) và `143256` (chủ yếu tĩnh, có đoạn xoay tay ngắn, 3 timeout), số mẫu `n=2` và `n=4` quá nhỏ và có lẫn chuyển động tay, nên cần chú thích rõ trong báo cáo khi trích dẫn.
3. **`Settling time = nan` (`settled = False` ở 100% các bước trong toàn bộ `steps.csv`):**
   - Trong [metrics.py:126-182](file:///home/dat/drone_tracking/pbl6_tricopter_vision/src/metrics.py#L126-L182), `YawStepAnalyzer` yêu cầu `|err_x| <= 15 px` liên tục ít nhất `0.5 s` trong cửa sổ `5.0 s`.
   - Trong toàn bộ 14 bước đo được ở các tệp `steps.csv`, `yaw_travel_deg` thực tế chỉ nằm trong khoảng `0.01° – 2.92°` (trừ 1 bước `11.02°` ở `143634` do bước nhảy khởi tạo yaw EKF từ `0.00°` sang `-11.07°` ở khung 44), tức thân drone không xoay vòng kín để đưa `err_x` về trong vùng chết ±15 px.

---

## Bảng 3: Cấu hình YAML thực tế & Trạng thái Phần cứng Raspberry Pi 5

### 3A. Cấu hình hợp nhất cho profile `raspberry_pi` (`config/common.yaml` + `config/raspberry_pi.yaml`)

*(Liệt kê cả cấu hình **hiện tại trên máy — dùng cho Đợt 2 ngày 19/11** và điểm khác biệt của **commit `HEAD` — dùng cho Đợt 1 ngày 17/11**)*

| Nhóm | Tham số | Giá trị hiệu lực hiện tại (Đợt 2 — 19/11) | Giá trị ở `HEAD` (Đợt 1 — 17/11) | Nguồn tệp |
|---|---|---|---|---|
| **Khung hình** | `frame.width x height` | `640 x 480` | `640 x 480` | `common.yaml` |
| | `frame.hfov` | `66.0°` | `66.0°` | `common.yaml` |
| **Camera** | `camera.fps` | `20` | `20` | `raspberry_pi.yaml` |
| | `camera.buffer_count` / `rotate_180` | `2` / `true` | `2` / `true` | `raspberry_pi.yaml` |
| | `camera.af_mode` / `lens_position` | `manual` / `0.3` D | `continuous` (`AfMode=2` hardcode) | `common.yaml` |
| | `camera.ae_exposure_mode` | `short` | `normal` (mặc định libcamera) | `common.yaml` |
| **Detector** | `detector.imgsz` | `640` | `640` | `common.yaml` |
| | `detector.conf_thresh` | `0.20` *(act: `0.45`, buf: `30`, match: `0.85`)* | `0.45` *(ByteTrack mặc định)* | `common.yaml` |
| | `detector.nms_thresh` | `0.45` | `0.45` | `common.yaml` |
| | `detector.ncnn_threads` | `2` | `2` | `common.yaml` |
| | `detector.ego_motion_comp` | `true` | *(chưa có)* | `common.yaml` |
| **Tracker / Control** | `tracker.yaw_kp` | `0.08` | `0.08` | `common.yaml` |
| | `tracker.yaw_kd` | `0.0` | `0.01` | `common.yaml` |
| | `tracker.yaw_deadband_px` | `15.0` *(exit: `26.0` px)* | `15.0` | `common.yaml` |
| | `tracker.max_yaw_rate` | `20.0` °/s *(accel: `60.0` °/s²)* | `25.0` °/s | `common.yaml` |
| | `tracker.min_vx` / `max_vx` | `0` / `0` *(đang khóa; comment: `0.15` / `1.2`)* | `0` / `0` *(riêng `114848` chạy lúc `max_vx=1.2`)* | `common.yaml` |
| | `tracker.latency_comp` / `coast_timeout` / `lost_timeout` | `true` / `0.6 s` / `3.0 s` | `false` / — / `2.0 s` | `common.yaml` |
| **MAVLink & Telemetry** | `mavlink.conn_str` / `baud` | `/dev/ttyAMA0` / `921600` | `/dev/ttyAMA0` / `921600` | `raspberry_pi.yaml` |
| | `mavlink.control_method` | `guided_vel` | `guided_vel` | `raspberry_pi.yaml` |
| | `telemetry.enabled` / `attitude_hz` | `true` / `50 Hz` | Chỉ bật khi `--debug` (30 Hz) | `common.yaml` |
| **Vòng lặp & Web** | `loop.target_fps` | `15.0` | `15.0` | `raspberry_pi.yaml` |
| | `loop.headless` / `auto_lock` | `true` / `true` *(ghi đè `common.yaml`)* | `true` / `true` | `raspberry_pi.yaml` |
| | `loop.hud_mode` / `terminal_log_interval` | `minimal` / `0.0 s` | `full` / `0.5 s` | `common.yaml` |
| | `web.enabled` / `port` / `jpeg_quality` | `true` / `5000` / `60` | `true` / `5000` / `60` | `raspberry_pi.yaml` |

### 3B. Thông tin Phần cứng & Trạng thái Hệ thống Raspberry Pi 5 (đo tại thời điểm kiểm tra)

| Hạng mục | Giá trị thực tế đọc từ máy |
|---|---|
| **Bo mạch (Model)** | `Raspberry Pi 5 Model B Rev 1.0` (kiến trúc `aarch64`, 4 nhân CPU) |
| **Bộ nhớ (RAM / Swap)** | `4.0 GiB` RAM (đang dùng `929 MiB`, khả dụng `3.0 GiB`) / `2.0 GiB` Swap (`0 B` sử dụng) |
| **Hệ điều hành & Kernel** | `Debian GNU/Linux 12 (bookworm)` — `Linux 6.12.109+rpt-rpi-2712 #1 SMP PREEMPT aarch64` |
| **Phiên bản Python** | `Python 3.11.2` (cả hệ thống và môi trường ảo `drone_env`) |
| **Nhiệt độ SoC (`vcgencmd measure_temp`)** | `49.4 °C` (trạng thái nghỉ khi kiểm tra) |
| **Điện áp lõi (`vcgencmd measure_volts core`)** | `0.8676 V` |
| **Trạng thái nguồn / sụt áp (`vcgencmd get_throttled`)** | `throttled=0x0` *(không có cờ sụt áp under-voltage, không giới hạn xung nhịp hay quá nhiệt trong phiên boot hiện tại)* |
