# 🛡️ Hệ Thống Giám Sát An Toàn Lao Động Thông Minh (AI Smart Safety Vision)

Hệ thống Computer Vision tích hợp Trí tuệ Nhân tạo (AI) giám sát an toàn lao động trong phân xưởng, công trường và khu công nghiệp:
- 👷 **Kiểm tra Trang thiết bị Bảo hộ (PPE)**: Phát hiện vi phạm Mũ bảo hộ (`Hardhat` / `NO-Hardhat`), Áo phản quang (`Safety Vest` / `NO-Safety Vest`), Khẩu trang (`Mask` / `NO-Mask`).
- 🚨 **Phát hiện Ngã & Bất động (Fall Detection)**: Sử dụng mô hình YOLOv8-Pose 17 khớp xương kết hợp kiểm tra động học chuyển động và trạng thái bất động sau va chạm.
- 🔥 **Phát hiện Cháy & Khói sớm (Fire & Smoke)**: Phân tích động học quang học và theo dõi chu kỳ đám cháy/khói thời gian thực.
- ⚡ **Pipeline Hai Giai Đoạn (Two-Stage Gating)**: Triệt tiêu 100% tình trạng khói cuộn hoặc vật thể nền bị nhận diện nhầm thành người hoặc vi phạm mồ côi.
- 🔊 **Cảnh báo Đa giác quan**: Còi hú âm thanh trực tiếp trên trình duyệt (Web Audio Engine) và hiển thị thông báo khẩn cấp.

---

## 📂 Cấu Trúc Thư Mục Dự Án

```text
ppe-detection-demo/
│
├── backend/                        # Máy chủ Backend (Python Flask + AI Models)
│   ├── app.py                      # Flask App Entrypoint & Socket.IO server
│   ├── camera_manager.py           # Quản lý luồng Camera/Webcam/RTSP
│   ├── config.py                   # Cấu hình hệ thống & đường dẫn lưu trữ
│   ├── detector.py                 # Module PPE & Pipeline 2 giai đoạn (Two-Stage Gating)
│   ├── fall_detector.py            # Giao diện bộ phát hiện ngã
│   ├── fire_detector.py            # Module nhận diện Khói & Lửa
│   ├── incident_lifecycle.py       # Quản lý vòng đời sự cố và cảnh báo
│   ├── models.py                   # SQLAlchemy Database Models (Lưu vi phạm, cấu hình)
│   ├── pose_fall_detector.py       # Bộ nhận diện tư thế người 17 keypoints YOLOv8-Pose
│   ├── train_ppe.py                # Script huấn luyện lại model PPE chuẩn công nghiệp
│   ├── requirements.txt            # Danh sách thư viện Python cần thiết
│   ├── models_dir/                 # Chứa các trọng số AI (Weights) đã tối ưu
│   │   ├── best_hardhat.pt         # Trọng số YOLOv8 phát hiện PPE
│   │   ├── yolov8s-pose.pt         # Trọng số YOLOv8-Pose 17 khớp xương người
│   │   ├── fire_smoke_yolov8n.pt   # Trọng số YOLOv8 phát hiện khói & lửa
│   │   └── fall_lstm.pth           # Model nơ-ron LSTM bổ trợ nhận diện ngã
│   ├── routes/                     # RESTful API Endpoints
│   │   ├── camera_routes.py        # API quản lý camera
│   │   ├── exclusion_routes.py     # API vùng loại trừ (Exclusion Zones)
│   │   ├── settings_routes.py      # API cài đặt hệ thống & độ nhạy
│   │   ├── stats_routes.py         # API thống kê số liệu
│   │   └── violation_routes.py     # API danh sách vi phạm & xuất bằng chứng
│   ├── tests/                      # Bộ kiểm thử tự động (Unit & Integration Tests)
│   └── static/                     # Thư mục lưu trữ bằng chứng, video upload & kết quả
│       ├── evidence/               # Ảnh chụp bằng chứng vi phạm
│       ├── outputs/                # Video đã được vẽ hộp nhận diện
│       └── uploads/                # Video tải lên từ người dùng
│
├── frontend/                       # Giao diện Web (React + Vite + Tailwind CSS)
│   ├── src/
│   │   ├── api/                    # Kết nối RESTful API Backend
│   │   ├── components/             # Các Component giao diện dùng chung
│   │   ├── pages/                  # Các trang chức năng chính
│   │   │   ├── Dashboard.jsx       # Trang tổng quan số liệu & biểu đồ
│   │   │   ├── Monitor.jsx         # Trang giám sát Camera / Webcam trực tiếp
│   │   │   ├── Upload.jsx          # Trang phân tích video tải lên
│   │   │   ├── Violations.jsx      # Trang nhật ký vi phạm & bằng chứng
│   │   │   └── Settings.jsx        # Trang cài đặt độ nhạy & ngưỡng cảnh báo
│   │   └── utils/
│   │       └── soundEngine.js      # Hệ thống phát âm thanh còi báo động Web Audio
│   ├── package.json                # Danh sách thư viện Frontend (NPM)
│   └── vite.config.js              # Cấu hình Vite & Proxy kết nối Backend
│
├── .gitignore                      # Cấu hình loại bỏ file tạm, video rác và cache
└── README.md                       # Tài liệu hướng dẫn cài đặt & vận hành
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Ứng Dụng

### 1. Yêu Cầu Môi Trường
- **Python**: Phiên bản 3.10 hoặc 3.11/3.12 (khuyến nghị cài kèm hỗ trợ CUDA nếu có GPU NVIDIA).
- **Node.js**: Phiên bản 18+ và **npm**.
- **Git**: Đã cài đặt trên máy.

---

### 2. Cài Đặt Backend

1. Mở terminal và chuyển vào thư mục `backend`:
   ```powershell
   cd backend
   ```
2. (Tùy chọn) Tạo môi trường ảo:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. Cài đặt các thư viện cần thiết:
   ```powershell
   pip install -r requirements.txt
   ```
4. Khởi động Backend Server:
   ```powershell
   python app.py
   ```
   *Backend sẽ khởi chạy tại: `http://localhost:5000`*

---

### 3. Cài Đặt Frontend

1. Mở một cửa sổ terminal mới và chuyển vào thư mục `frontend`:
   ```powershell
   cd frontend
   ```
2. Cài đặt các gói phụ thuộc:
   ```powershell
   npm install
   ```
3. Khởi chạy giao diện phát triển:
   ```powershell
   npm run dev
   ```
   *Frontend sẽ khởi chạy tại: `http://localhost:5173`*

---

### 4. Sử Dụng Hệ Thống
1. Mở trình duyệt truy cập: **`http://localhost:5173`**.
2. **Tab Giám sát (Monitor)**:
   - Chọn nguồn Camera (Webcam máy tính hoặc nhập địa chỉ luồng RTSP của Camera IP xưởng).
   - Bật/tắt các tính năng nhận diện: PPE, Phát hiện Ngã, Phát hiện Khói/Lửa.
3. **Tab Phân tích Video (Upload)**:
   - Tải lên video hiện trường (hỗ trợ `.mp4`, `.avi`, `.mov`).
   - Xem thanh tiến trình phân tích theo thời gian thực và xem video kết quả có gắn khung nhận diện.
4. **Tab Cài đặt (Settings)**:
   - Tinh chỉnh các ngưỡng tin cậy (Confidence Thresholds) và bật/tắt chuông cảnh báo âm thanh.

---

## 🛠️ Huấn Luyện Lại (Retraining) Model PPE
Nếu bạn muốn bổ sung dữ liệu mới và huấn luyện lại model PPE để triệt tiêu hoàn toàn báo động giả:
```powershell
cd backend
python train_ppe.py --data path/to/dataset/data.yaml --epochs 100 --extract-video-bg static/uploads/video_smoke.mp4
```
*Script sẽ tự động nạp ảnh nền âm tính (khói, lửa, xưởng trống) vào dataset, huấn luyện với Early Stopping, tự backup model cũ và cập nhật weights mới vào `backend/models_dir/best_hardhat.pt`.*
