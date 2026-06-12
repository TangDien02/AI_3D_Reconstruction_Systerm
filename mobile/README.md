# 3DRecon Mobile App (Expo & React Native)

Đây là ứng dụng di động trong hệ thống **AI 3D Reconstruction**, cho phép người dùng chụp ảnh vật thể và biến chúng thành mô hình 3D (Mesh) có đầy đủ vân bề mặt (Texture).

## Tính năng chính

- **Live Scan (YOLO):** Tự động phát hiện vật thể thời gian thực qua camera bằng mô hình YOLO.
- **Manual Crop:** Cho phép người dùng tự vẽ vùng chọn (Bounding Box) quanh vật thể để tái tạo.
- **Visual Processing Timeline:** Hiển thị tiến trình xử lý từng bước (Capture -> Clean -> Mesh -> Texture).
- **Interactive 3D Viewer:** Xem mô hình 3D ngay trong ứng dụng với các chế độ:
  - Xoay/Phóng to (Orbit Controls).
  - Chế độ lưới (Wireframe mode).
  - Tự động xoay (Auto-rotate).
- **History Gallery:** Lưu trữ lịch sử các bản quét cục bộ trên điện thoại bằng `AsyncStorage`.
- **Hỗ trợ Hunyuan3D-2:** Kết nối mạnh mẽ với backend để xử lý mô hình 3D chất lượng cao.

## Cấu trúc thư mục (Sau khi Refactor)

```text
mobile/
├── src/
│   ├── components/
│   │   ├── LogoMark.js           # Component hiển thị logo ứng dụng
│   │   ├── ProcessingTimeline.js # Thanh tiến trình xử lý 4 bước
│   │   └── Viewer3DModal.js      # Trình xem 3D tích hợp Three.js qua WebView
│   ├── theme.js                  # Quản lý bảng màu (C), API URL và Config hệ thống
│   └── utils.js                  # Các hàm bổ trợ (Padding bbox, Error handling, Delay)
├── App.js                        # Luồng điều khiển chính (State & API Actions)
├── package.json                  # Quản lý thư viện (Expo 54, Three.js, WebView)
└── README.md                     # Tài liệu hướng dẫn này
```

## 🛠 Hướng dẫn thiết lập

### 1. Cài đặt thư viện
Tại thư mục `mobile/`, chạy lệnh:
```bash
npm install
```

### 2. Cấu hình API Backend
Mở file `src/theme.js` hoặc tạo file `.env.local` để cấu hình địa chỉ Backend của bạn:
```javascript
// src/theme.js
export const API_BASE_URL = "http://<IP_CỦA_BẠN>:8000";
```

### 3. Chạy ứng dụng
```bash
npx expo start
```
Sử dụng ứng dụng **Expo Go** (trên iOS/Android) để quét mã QR và trải nghiệm.

## Lưu ý dành cho lập trình viên (Code Reader)

1.  **Luồng xử lý (Data Flow):**
    - Camera chụp ảnh -> Gửi `FormData` lên `/reconstruct-bbox` hoặc `/reconstruct-object`.
    - Backend trả về `job_id`.
    - App thực hiện **Polling** (gọi API liên tục mỗi 5s) qua hàm `waitForReconstructionJob` cho đến khi trạng thái là `done`.
2.  **Xử lý Bounding Box:**
    - Để tránh vật thể bị cắt mất khi tách nền, chúng tôi luôn thêm **Padding (10-15%)** vào vùng chọn trước khi gửi lên Server (xem hàm `addPaddingToBbox` trong `utils.js`).
3.  **Trình xem 3D:**
    - Sử dụng `WebView` để render một trang HTML chứa thư viện **Three.js** và **GLTFLoader**. Điều này giúp app nhẹ hơn và dễ dàng tùy chỉnh shader/ánh sáng.
4.  **Lưu trữ:**
    - Dữ liệu lịch sử được lưu dưới dạng JSON trong `AsyncStorage`. Các file 3D thực tế vẫn nằm trên Server, app chỉ lưu đường dẫn (Path).

---
*Phát triển bởi AI 3D Reconstruction Team.*
