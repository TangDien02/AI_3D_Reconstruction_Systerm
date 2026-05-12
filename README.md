# 🚀 AI 3D Reconstruction System (ReconApp)

Dự án này là một hệ thống hoàn chỉnh (End-to-End) dùng để **tái tạo mô hình 3D của vật thể từ ảnh 2D hoặc video quét 360 độ**. 
Hệ thống bao gồm một ứng dụng di động (Frontend) để thu thập dữ liệu trực quan và một máy chủ AI (Backend) để phân tích thị giác máy tính và dựng 3D.

---

## 🏗️ Kiến trúc Hệ thống

Dự án được chia thành hai luồng xử lý chính:

### 1. 📱 Mobile App (React Native / Expo)
Đóng vai trò là Client thu thập dữ liệu, hướng dẫn người dùng và hiển thị kết quả.

- **Công nghệ:** React Native, Expo Go, TypeScript, `@testing-library/react-native`.
- **Thư mục chính (`/src`):**
  - `screens/`: Chứa các màn hình chính như `IntroScreen` (giới thiệu luồng làm việc) và `ScannerScreen` (giao diện mở Camera, xin quyền, và hiển thị lớp phủ UI quét vật thể).
  - `components/`: Các thành phần tái sử dụng như `PrimaryButton`, `ScanFrame`, `PipelineItem`.
  - `constants/`: Định nghĩa giao diện (`theme.ts`) và quy trình workflow (`workflow.ts`).
  - `types/`: Chứa định nghĩa TypeScript (`app.ts`) cho toàn bộ app.
- **Tính năng nổi bật:** Xin quyền Camera tự động, UI thân thiện, hỗ trợ Automation Test (Jest) với cấu trúc DOM ảo (kịch bản test tại `__tests__/App.test.tsx`).

### 2. 🖥️ AI Backend Server (Python / FastAPI)
Đóng vai trò là "Bộ não" xử lý ảnh, Tracking và tái tạo không gian 3D.

- **Công nghệ:** FastAPI, Uvicorn, PyTorch, Docker.
- **Thư mục chính (`/server`):**
  - `main.py`: Khởi tạo các API Endpoints (`/detect-frame`, `/upload-scan-video`, `/scan-status/{job_id}`).
  - `vit_reconstruction.py`: Pipeline Deep Learning cốt lõi sử dụng **Vision Transformer (ViT)**. Xử lý ảnh (Patch Extraction), chạy qua các khối Multi-Head Self-Attention để trích xuất đặc trưng không gian (3D Features) phục vụ cho Photogrammetry hoặc các hệ thống trích xuất Voxel/Point Cloud.
  - `Dockerfile` & `.dockerignore`: Cấu hình đóng gói hệ thống backend siêu nhẹ, tối ưu hóa các biến môi trường của Python.

---

## ⚙️ Workflow Luồng Xử Lý (End-to-End)

1. **[ READY ]**: Người dùng mở ứng dụng, vào màn hình `IntroScreen`.
2. **[ DETECTING ]**: Chuyển sang `ScannerScreen`, Camera bật. Điện thoại bắt đầu gửi các frame ảnh lên Server qua API `/detect-frame`. Server dùng YOLO để trả về Bounding Box khoanh vùng vật thể.
3. **[ OBJECT_SELECTED ]**: Người dùng chạm vào Bounding Box để xác nhận vật thể muốn quét 3D.
4. **[ SCANNING ]**: Người dùng quay video 360 độ xung quanh vật thể. Ứng dụng khóa nét và hỗ trợ tracking để người dùng quay đều mọi góc độ.
5. **[ UPLOADING ]**: Ứng dụng gửi video quay được lên Server.
6. **[ PROCESSING ]**: Server trích xuất frame, tách nền, và chạy qua mạng nơ-ron **ViT 3D Engine** (`vit_reconstruction.py`) để dựng hình.
7. **[ DONE ]**: Trả về đường dẫn chứa file 3D (`.glb` / `.obj`) cho ứng dụng di động hiển thị.

> 🎮 **Mô phỏng trực quan:** Bạn có thể mở file `pixel_simulation.html` ở thư mục gốc bằng trình duyệt web để xem hình ảnh mô phỏng UI phong cách Pixel Art chạy đúng theo Workflow 7 bước phía trên!

---

## 🛠️ Hướng dẫn cài đặt và chạy dự án

### 1. Khởi động Mobile App
```bash
# Cài đặt thư viện
npm install

# Chạy ứng dụng bằng Expo
npm start
```

### 2. Khởi động Backend Server
```bash
cd server

# Cài đặt thư viện Python
pip install -r requirements.txt

# Chạy FastAPI Server
uvicorn main:app --host 0.0.0.0 --port 8000
```
*(Hoặc dùng Docker)*
```bash
cd server
docker build -t 3drecon-server .
docker run -p 8000:8000 3drecon-server
```

### 3. Chạy Unit Test (DOM Simulation)
Dự án đã được tích hợp Jest để test giao diện tự động. Quá trình test không cần bật máy ảo.
```bash
npm test
```

---

## 📈 Lộ trình phát triển tiếp theo (Phase 2)
- Thay thế API Polling bằng WebSockets cho API `/detect-frame` để giảm thiểu hoàn toàn độ trễ (latency).
- Tích hợp thư viện hiển thị vật thể 3D (ví dụ: `react-native-webview` kết hợp Three.js, hoặc Expo GL) để render model `.glb` trực tiếp trên app.
- Nâng cấp cơ chế Upload Video thành Chunked Upload để tránh lỗi mạng khi upload video dung lượng lớn.
