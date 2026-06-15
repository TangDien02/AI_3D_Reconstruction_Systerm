# 3D Object Reconstruction Core (ResNet50 + MLP)

Đây là bộ mã nguồn lõi đã được đóng gói, tập trung vào mô hình baseline ResNet50 + MLP cho bài toán tái tạo vật thể 3D từ ảnh đơn (Single-view 3D Reconstruction).

## 🚀 Cấu trúc dự án

Toàn bộ mã nguồn nằm trong thư mục `project/`:

- `project/src/models`: ResNet Encoder & Point Cloud Decoder.
- `project/src/metrics`: Chamfer Distance, F-Score, và các hàm loss.
- `project/src/evaluation`: Pipeline đánh giá mô hình.
- `project/src/data`: Dataloader cho Pix3D dataset.
- `project/src/training`: Pipeline huấn luyện mô hình.
- `project/notebooks`: Notebook hướng dẫn chi tiết.

## 🛠️ Cài đặt

1. Di chuyển vào thư mục project:
   ```bash
   cd project
   ```

2. Cài đặt các thư viện phụ thuộc:
   ```bash
   pip install -r requirements.txt
   ```

## 📖 Hướng dẫn sử dụng

Xem notebook hướng dẫn chi tiết tại:
`project/notebooks/RECONSTRUCTION_BASELINE_GUIDE.ipynb`

### Các lệnh chính (Chạy từ thư mục `project/`):

- **Tiền xử lý dữ liệu:**
  ```bash
  python -m src.preprocessing.build_processed_dataset
  ```

- **Huấn luyện mô hình:**
  ```bash
  python train.py --categories chair --epochs 100 --batch-size 8
  ```

- **Đánh giá mô hình:**
  ```bash
  python -m src.evaluation.evaluate_baseline --checkpoint results/chair_resnet_baseline/outputs/checkpoints/best_model.pt
  ```

## 🎯 Thành phần lõi

- **Encoder:** ResNet50 (Pretrained on ImageNet).
- **Decoder:** MLP (Multi-Layer Perceptron) dự đoán tọa độ N điểm.
- **Loss Function:** Weighted Chamfer Distance.
- **Evaluation:** F-Score @ threshold.
