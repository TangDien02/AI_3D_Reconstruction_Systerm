# Project Audit - 3D Reconstruction Pipeline

Tài liệu này mô tả trạng thái hiện tại của folder `project/`: nhiệm vụ từng thư mục, chức năng từng file, các hàm/class chính, thuật toán đang dùng và các điểm chưa khớp với workflow end-to-end.

## 1. Vai Trò Của Folder `project/`

`project/` là phần ML chính của repo. Luồng hiện tại là:

```text
Pix3D raw data
  -> clean metadata
  -> crop/mask/resize images
  -> sample CAD mesh thành point cloud
  -> train TransformerPointCloudNet
  -> evaluate bằng Chamfer Distance/F-score
  -> inference ảnh đơn thành point cloud
```

Đầu ra hiện là **point cloud** (`.npy`, `.ply`, `.png preview`), chưa phải mesh hoàn chỉnh `.obj`/`.glb`.

## 2. Nhiệm Vụ Theo Folder

| Folder | Nhiệm vụ |
| --- | --- |
| `data/processed/` | Chứa metadata sạch, split train/val/test và artifact processed nếu đã chạy preprocessing. |
| `notebooks/` | Notebook EDA và baseline preprocessing/training phục vụ phân tích/báo cáo. |
| `report/` | Ghi chú/báo cáo kỹ thuật cho baseline. |
| `src/data/` | Dataset loader cho Pix3D raw và processed. |
| `src/preprocessing/` | Làm sạch metadata, crop/mask ảnh, sample point cloud từ mesh. |
| `src/models/` | Kiến trúc model học ảnh 2D -> point cloud 3D. |
| `src/training/` | Vòng lặp train, validate, log metric, lưu checkpoint. |
| `src/evaluation/` | Evaluate checkpoint trên split đã xử lý. |
| `src/inference/` | Inference ảnh đơn và so sánh point cloud dự đoán/ground truth. |
| `src/metrics/` | Chamfer Distance, F-score và metric mesh baseline. |
| `src/pipeline/` | Baseline template pipeline cũ: lấy class ảnh rồi chọn mesh template. |
| `src/utils/` | Logger, lưu point cloud, biểu đồ/visualization. |

## 3. File Và Function/Class Chính

### Root `project/`

| File | Hàm/Class | Nhiệm vụ |
| --- | --- | --- |
| `main_workflow.py` | `parse_args` | Định nghĩa tham số workflow tổng. |
| `main_workflow.py` | `run_preprocessing` | Gọi clean metadata, save splits, build image/mask, build point cloud. |
| `main_workflow.py` | `make_training_args` | Map tham số workflow sang tham số training. |
| `main_workflow.py` | `ensure_training_dependencies` | Kiểm tra dependency tối thiểu trước khi train. |
| `main_workflow.py` | `main` | Điều phối preprocessing -> training. |
| `train.py` | `main` từ `src.training.training_pipeline` | Entry rút gọn để train baseline. |
| `TECHNICAL_COMMANDS.md` | N/A | Tập lệnh chạy preprocessing, training, evaluation, inference, backend. |

### `src/data/dataloader.py`

| Hàm/Class | Nhiệm vụ |
| --- | --- |
| `normalize_points` | Chuẩn hóa point cloud: trừ tâm và scale theo khoảng cách lớn nhất. |
| `load_and_sample_obj` | Load mesh `.obj`, sample surface thành point cloud tensor. |
| `Pix3DDataset` | Dataset đọc Pix3D raw: ảnh, mask, model, category. |
| `Pix3DDataset.__getitem__` | Apply mask, resize ảnh, convert tensor, sample ground-truth points từ mesh. |
| `ProcessedPix3DDataset` | Dataset đọc split CSV processed, ảnh processed và point cloud `.npy`. |
| `ProcessedPix3DDataset.__getitem__` | Load ảnh/tensor và point cloud ground truth cho training/evaluation. |

### `src/preprocessing/metadata_cleaner.py`

| Hàm/Class | Nhiệm vụ |
| --- | --- |
| `load_pix3d_json` | Đọc `pix3d.json` thành DataFrame. |
| `_relative_processed_image_path` | Sinh path ảnh processed theo category. |
| `_relative_processed_mask_path` | Sinh path mask processed theo category. |
| `_relative_point_path` | Sinh path point cloud `.npy` theo model/category. |
| `clean_pix3d_metadata` | Lọc record thiếu cột/file, lọc category, tạo `sample_id` và path processed. |
| `make_stratified_splits` | Chia train/val/test theo từng category. |
| `save_metadata_and_splits` | Lưu metadata sạch và split CSV. |
| `_safe_to_csv` | Ghi CSV qua file tạm để giảm lỗi file lock. |
| `parse_args`, `main` | CLI làm sạch metadata và tạo split. |

### `src/preprocessing/build_processed_dataset.py`

| Hàm/Class | Nhiệm vụ |
| --- | --- |
| `parse_bbox` | Chuyển bbox từ list/string sang crop box hợp lệ theo kích thước ảnh. |
| `preprocess_image_and_mask` | Crop ảnh/mask theo bbox, apply mask, resize và lưu PNG. |
| `load_image_safely` | Đọc ảnh, xử lý palette transparency, convert mode. |
| `build_processed_images` | Lặp qua metadata để tạo ảnh/mask processed. |
| `parse_args`, `main` | CLI build metadata, ảnh/mask và point cloud. |

### `src/preprocessing/mesh_processor.py`

| Hàm/Class | Nhiệm vụ |
| --- | --- |
| `normalize_points` | Chuẩn hóa point cloud từ mesh. |
| `load_mesh` | Load mesh bằng `trimesh`, gộp scene nếu cần, chặn mesh rỗng. |
| `sample_mesh_points` | Sample điểm trên bề mặt mesh bằng `trimesh.sample.sample_surface`. |
| `save_pointcloud` | Lưu point cloud `.npy`, có option bỏ qua nếu file đã tồn tại. |
| `build_pointclouds_from_metadata` | Tạo point cloud cho các model duy nhất trong metadata. |
| `parse_args`, `main` | CLI sample point cloud từ metadata. |

### `src/preprocessing/image_processor.py`

| Hàm/Class | Nhiệm vụ |
| --- | --- |
| `ImagePreprocessor` | Wrapper phân loại ảnh: ưu tiên metadata, fallback YOLOv8 segmentation nếu cài được `ultralytics`. |
| `get_class_from_pre_data` | Lấy class từ metadata truyền vào. |
| `process` | Trả class từ pre-data hoặc chạy YOLO để lấy class của box đầu tiên. |

### `src/models/transformer_pointcloud.py`

| Hàm/Class | Nhiệm vụ |
| --- | --- |
| `TransformerPointCloudNet` | Model chính: ảnh RGB -> point cloud `[num_points, 3]`. |
| `TransformerPointCloudNet.forward` | Patch embedding bằng Conv2D, thêm class/position token, Transformer encoder, MLP decoder ra tọa độ 3D. |

### `src/training/training_pipeline.py`

| Hàm/Class | Nhiệm vụ |
| --- | --- |
| `setup_baseline_logger` | Ghi log ra file và console. |
| `save_training_curves` | Vẽ loss/metric theo epoch. |
| `train_one_epoch` | Train một epoch với Chamfer Distance và Adam. |
| `evaluate` | Validate model bằng Chamfer Distance và F-score. |
| `parse_args` | CLI training baseline. |
| `build_checkpoint` | Đóng gói model state + hyperparameter + metric. |
| `is_better_score` | So sánh checkpoint tốt hơn theo metric được chọn. |
| `run_training` | Điều phối dataset, dataloader, model, optimizer, train loop, metric CSV, checkpoint, summary. |
| `main` | Entry CLI. |

### `src/evaluation/evaluate_baseline.py`

| Hàm/Class | Nhiệm vụ |
| --- | --- |
| `evaluate_checkpoint` | Load checkpoint, chạy trên split processed, lưu metric theo batch và summary JSON. |
| `parse_args`, `main` | CLI evaluation. |

### `src/inference/baseline_inference.py`

| Hàm/Class | Nhiệm vụ |
| --- | --- |
| `load_image_tensor` | Load ảnh RGB, resize, chuẩn hóa `[0, 1]`, đổi sang tensor `[1, 3, H, W]`. |
| `load_baseline_model` | Load checkpoint và dựng đúng cấu hình `TransformerPointCloudNet`. |
| `predict_pointcloud` | Inference ảnh đơn thành numpy point cloud. |
| `parse_args`, `main` | CLI inference, lưu `.npy`, `.ply`, `.png`, summary JSON. |

### `src/inference/compare_pointclouds.py`

| Hàm/Class | Nhiệm vụ |
| --- | --- |
| `load_checkpoint_model` | Load checkpoint và model cho so sánh. |
| `sample_points` | Giảm số điểm khi vẽ để hình nhẹ hơn. |
| `set_equal_3d_axes` | Đặt tỉ lệ trục 3D cân bằng. |
| `save_comparison_figure` | Lưu hình so sánh predicted vs ground truth point cloud. |
| `compare_sample` | Chọn một sample, predict, tính metric, lưu `.npy` và hình comparison. |
| `parse_args`, `main` | CLI so sánh point cloud. |

### `src/metrics/losses.py`

| Hàm/Class | Nhiệm vụ |
| --- | --- |
| `chamfer_distance` | Tính symmetric Chamfer Distance bằng `torch.cdist`. |
| `f_score` | Tính F-score, precision, recall theo ngưỡng khoảng cách. |

### `src/metrics/evaluator.py`

| Hàm/Class | Nhiệm vụ |
| --- | --- |
| `sample_points_from_mesh` | Sample point cloud từ mesh để đánh giá template baseline. |
| `compute_metrics` | Tính Chamfer Distance và F-score giữa hai mesh bằng KDTree. |

### `src/pipeline/baseline_runner.py`

| Hàm/Class | Nhiệm vụ |
| --- | --- |
| `BaselinePipeline` | Pipeline cũ/template: lấy class ảnh, chọn mesh template, đánh giá với ground truth. |
| `run_single` | Chạy một ảnh: classify -> lấy template mesh -> tính metric. |

### `src/utils/logger.py`

| Hàm/Class | Nhiệm vụ |
| --- | --- |
| `get_logger` | Tạo logger console format thống nhất. |

### `src/utils/pointcloud_io.py`

| Hàm/Class | Nhiệm vụ |
| --- | --- |
| `ensure_pointcloud_array` | Validate point cloud có shape `[N, 3]` và không rỗng. |
| `save_pointcloud_npy` | Lưu point cloud dạng `.npy`. |
| `save_pointcloud_ply` | Lưu point cloud dạng ASCII `.ply`. |

### `src/utils/visualization.py`

| Hàm/Class | Nhiệm vụ |
| --- | --- |
| `_prepare_output_path`, `_save_current_figure` | Chuẩn bị path và lưu figure matplotlib. |
| `plot_category_distribution` | Vẽ phân bố category. |
| `plot_cleaning_comparison` | Vẽ số mẫu trước/sau cleaning. |
| `plot_missing_files` | Vẽ số file thiếu. |
| `plot_image_size_distribution` | Vẽ scatter kích thước ảnh. |
| `plot_baseline_metrics` | Vẽ metric baseline dạng bar chart. |
| `plot_point_cloud` | Vẽ point cloud 3D. |
| `save_dataset_summary_tables` | Lưu bảng summary dataset và class distribution. |
| `save_week2_visualizations` | Sinh bộ biểu đồ/bảng cho báo cáo tuần 2. |

## 4. Thuật Toán Và Kỹ Thuật Được Sử Dụng

| Nhóm | Thuật toán/kỹ thuật | File liên quan |
| --- | --- | --- |
| Data cleaning | Lọc metadata, kiểm tra file tồn tại, lọc category | `metadata_cleaner.py` |
| Data split | Stratified train/val/test split theo category | `metadata_cleaner.py` |
| Image preprocessing | Bbox crop, binary mask, background trắng, resize bilinear/nearest | `build_processed_dataset.py`, `dataloader.py` |
| 3D preprocessing | Surface sampling từ mesh CAD | `mesh_processor.py`, `dataloader.py` |
| Normalization | Centering + scale point cloud về khoảng chuẩn | `mesh_processor.py`, `dataloader.py` |
| Model | Vision Transformer-style encoder: patch embedding, class token, positional embedding, Transformer encoder | `transformer_pointcloud.py` |
| Decoder | MLP decode embedding thành `num_points * 3`, activation `Tanh` | `transformer_pointcloud.py` |
| Optimization | Adam optimizer, backpropagation | `training_pipeline.py` |
| Loss | Symmetric Chamfer Distance | `losses.py` |
| Metrics | Chamfer Distance, F-score, precision, recall | `losses.py`, `evaluate_baseline.py` |
| Mesh metric baseline | KDTree nearest-neighbor distance | `metrics/evaluator.py` |
| Visualization | Matplotlib 2D/3D plots | `utils/visualization.py`, `compare_pointclouds.py` |
| Optional detection | YOLOv8-seg fallback for class extraction | `image_processor.py` |

## 5. Các Điểm Chưa Khớp/Cần Tối Ưu

1. **Single-view vs video workflow**: code ML chính xử lý một ảnh, trong khi README cũ mô tả video 360 độ/multi-view.
2. **Point cloud vs mesh output**: model xuất point cloud, chưa reconstruct mesh surface hoặc export `.glb`.
3. **`server/vit_reconstruction.py` chưa nối training thật**: file này minh họa ViT riêng, không dùng checkpoint `TransformerPointCloudNet`.
4. **Mock API còn nhiều**: detection/video/status chưa xử lý thật.
5. **Test frontend thiếu target**: `__tests__/App.test.tsx` import `../App`, nhưng thiếu `App.tsx`.
6. **Test API blog không cùng domain**: `tester/` đang kiểm thử CRUD blog, nên nên đổi sang test `/health`, `/detect-frame`, `/reconstruct-image`.
7. **Checkpoint path không thống nhất ở một vài command**: nên ưu tiên `results/baseline/outputs/checkpoints/transformer_pointcloud_net.pt`.
8. **Dữ liệu processed chỉ có CSV trong repo**: ảnh/mask/pointcloud processed có thể chưa được commit, cần chạy preprocessing trước khi train/inference.

## 6. Đề Xuất Tích Hợp Tiếp

1. Tạo lại `App.tsx`/`index.ts` hoặc bỏ hẳn test frontend nếu phạm vi chỉ là backend/ML.
2. Viết test backend đúng domain: health, detect mock, checkpoint missing, reconstruct-image khi có checkpoint.
3. Hợp nhất `server/vit_reconstruction.py` với `project/src/models/transformer_pointcloud.py`, hoặc chuyển file này vào `research/` để tránh hiểu nhầm.
4. Nếu cần video 360 thật: thêm frame extraction, object tracking/segmentation, multi-view fusion hoặc photogrammetry/NeRF/3D Gaussian Splatting pipeline.
5. Nếu cần mesh `.glb`: thêm bước point cloud -> mesh reconstruction như Poisson surface reconstruction hoặc Ball Pivoting, rồi export qua `trimesh`.
