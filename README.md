# 3D Object Reconstruction từ Ảnh 2D

Dự án này nghiên cứu và triển khai bài toán **tái tạo mô hình 3D của vật thể từ ảnh 2D** sử dụng các bộ dữ liệu chuẩn như ShapeNet, Pix3D, ScanNet và đánh giá bằng các thước đo hình học chuyên biệt (Chamfer Distance, F-score, IoU).[web:127][web:11]

## 1. Mục tiêu

- Xây dựng mô hình học sâu nhận đầu vào là ảnh (single-view hoặc multi-view) và sinh ra biểu diễn 3D (point cloud/voxel/mesh).
- So sánh hiệu năng trên các bộ dữ liệu benchmark: ShapeNet, Pix3D, ScanNet.
- Đánh giá mô hình bằng Chamfer Distance, F-score theo ngưỡng khoảng cách, và IoU cho voxel occupancy.

## 2. Bài toán

- **Đầu vào**: Ảnh RGB của vật thể (có thể kèm depth hoặc mask), được căn chỉnh từ các dataset.
- **Đầu ra**:
  - Point cloud 3D hoặc voxel grid biểu diễn hình dạng vật thể.
  - Có thể chuyển đổi sang mesh để trực quan hóa.
- **Yêu cầu**:
  - Giữ được hình dáng tổng thể, tỉ lệ và chi tiết bề mặt.
  - Tái tạo đủ bề mặt (không bị thiếu mảng lớn, ít lỗ hổng).

## 3. Bộ dữ liệu

Hiện tại dự án hỗ trợ các dataset dưới đây (có thể bật/tắt ở file config):

- **ShapeNet**: Kho mô hình CAD 3D synthetic với nhiều hạng mục (ghế, bàn, máy bay, v.v.), dùng cho training và evaluation cơ bản trên dữ liệu sạch.[web:77]
- **Pix3D**: Ảnh thực + mô hình CAD được gán thẳng hàng pixel-level, dùng để kiểm tra model trên dữ liệu thực.[web:81][web:87]
- **ScanNet (tùy chọn)**: RGB-D indoor scenes cùng mesh reconstruction, dùng cho thí nghiệm mở rộng ở môi trường thực phức tạp.[web:79][web:73]

Trong mã nguồn, đường dẫn dữ liệu được cấu hình trong `configs/datasets.yaml`. Tham khảo README của từng dataset để tải về và giải nén đúng cấu trúc thư mục.[web:127]

## 4. Kiến trúc mô hình

- Encoder 2D trích xuất đặc trưng từ ảnh (ResNet/ViT).
- Decoder 3D ánh xạ latent vector sang point cloud/voxel 3D.
- Loss function kết hợp giữa:
  - Chamfer Distance giữa point cloud dự đoán và ground-truth.
  - Regularization (ví dụ: smoothness, occupancy loss nếu là voxel).

(Phần này bạn chỉnh lại đúng với kiến trúc cụ thể: PointNet-based decoder, implicit field, NeRF, v.v.)

## 5. Thước đo đánh giá

Vì đầu ra là hình dạng 3D liên tục, dự án **không sử dụng Accuracy phân loại đơn giản**, mà dùng các metric sau:

- **Chamfer Distance (CD)**
  - Đo khoảng cách trung bình hai chiều giữa point cloud dự đoán và ground-truth.
  - Giá trị càng nhỏ nghĩa là hình dạng dự đoán càng gần với vật thể thật.[web:21][web:11]

- **F-score ở ngưỡng d**
  - Xem một điểm là “đúng” nếu khoảng cách đến bề mặt đối phương < d.
  - Tính Precision, Recall và F-score để phản ánh **độ chính xác** và **độ bao phủ** của reconstruction.[web:128][web:116]

- **IoU (Intersection over Union)**
  - Áp dụng khi biểu diễn dạng voxel hoặc occupancy.
  - Đo mức độ trùng khớp thể tích giữa reconstruction và ground-truth.[web:22][web:117]

**Lý do không dùng “Accuracy” đơn giản**:

- Bài toán không phải “đoán nhãn” mà là khớp hình dạng liên tục trong không gian 3D.
- Một con số Accuracy không thể hiện được vật thể có bị méo, bị thiếu vùng, hay bề mặt xấu.
- CD, F-score và IoU được dùng rộng rãi trong các nghiên cứu 3D reconstruction hiện đại để đánh giá cả **độ chính xác hình học** lẫn **độ hoàn chỉnh bề mặt**.[web:22][web:11][web:128]

## 6. Cài đặt

```bash
# 1. Tạo môi trường
conda create -n 3drecon python=3.10
conda activate 3drecon

# 2. Cài đặt dependencies
pip install -r requirements.txt

# 3. Thiết lập đường dẫn dữ liệu trong configs/datasets.yaml
```

## 7. Huấn luyện

Ví dụ huấn luyện trên ShapeNet:

```bash
python train.py \
  --config configs/shapenet_pointcloud.yaml \
  --exp_name shapenet_cd_fscore
```

- Logs và checkpoints sẽ được lưu trong `runs/shapenet_cd_fscore/`.
- TensorBoard có thể được bật qua:

```bash
tensorboard --logdir runs
```

## 8. Đánh giá

Sau khi huấn luyện, chạy:

```bash
python eval.py \
  --config configs/shapenet_pointcloud.yaml \
  --checkpoint runs/shapenet_cd_fscore/best.ckpt
```

Script sẽ xuất các metric: Chamfer Distance, F-score (với các ngưỡng khoảng cách), và IoU (nếu dùng voxel). File kết quả được lưu dưới dạng CSV trong `results/`.[web:117]

## 9. Visualize kết quả

```bash
python visualize.py \
  --config configs/shapenet_pointcloud.yaml \
  --checkpoint runs/shapenet_cd_fscore/best.ckpt
```

- Hiển thị side-by-side: ảnh input, point cloud/mesh ground-truth, và reconstruction.
- Có thể lưu ra `.ply` hoặc `.obj` để mở trong MeshLab/Blender.

## 10. Tài liệu tham khảo

- ShapeNet: _An Information-Rich 3D Model Repository_.[web:77]
- Pix3D: _Dataset and Methods for Single-Image 3D Shape Modeling_.[web:81]
- Tổng hợp metric đánh giá 3D reconstruction và mapping.[web:22][web:109][web:110]
