# AI 3D Reconstruction System

Repository này tập trung vào bài toán **3D reconstruction từ ảnh 2D**, với hướng phát triển từ baseline **single-object reconstruction** sang pipeline đầy đủ hơn cho **multi-object scene reconstruction**.

Mục tiêu dài hạn của hệ thống:

```text
Scene image / ảnh thực tế
→ tìm và tách từng object
→ chuẩn hóa object crop
→ reconstruct 3D từng object
→ xuất object set gồm label/unknown + bbox/mask + point cloud/mesh + metadata
```

Trạng thái code hiện tại trong `project/` đang chạy chắc nhất ở nhánh:

```text
Pix3D object image
→ preprocessing bbox/mask/resize
→ point cloud ground truth từ CAD mesh
→ ResNet encoder + point cloud decoder
→ point cloud prediction
→ Chamfer Distance / F-score
```

Roadmap kỹ thuật đề xuất tiếp theo nằm ở [project/PIPELINE_ROADMAP.md](project/PIPELINE_ROADMAP.md).

## 1. Bài Toán Và Output

Hệ thống được thiết kế theo hai mức:

```text
Mức 1 - Object-level reconstruction:
Một ảnh object đã crop sẵn → point cloud / mesh / voxel của object đó.

Mức 2 - Scene-level reconstruction:
Một ảnh/cảnh có nhiều object → tập các object 3D riêng lẻ.
```

Output mong muốn ở mức scene:

```json
[
  {
    "scene_id": "scene_001",
    "object_id": "obj_001",
    "label": "chair",
    "label_status": "predicted",
    "confidence": 0.92,
    "bbox": [120, 80, 260, 310],
    "mask_path": "masks/obj_001.png",
    "pointcloud_path": "outputs/obj_001.ply",
    "domain": "real"
  }
]
```

## 2. Cấu Trúc Chính

```text
.
├── project/                 # Pipeline ML: preprocessing, training, evaluation, inference
├── server/                  # FastAPI backend: mock scan API và image-to-pointcloud inference
├── tester/                  # Bộ test API hiện còn lệch domain, chưa phải test 3D chính
├── package.json             # Cấu hình npm/Jest cũ, frontend source hiện chưa đầy đủ
├── docker-compose.yml       # Backend + service phụ trợ
└── README.md
```

Các tài liệu quan trọng:

```text
project/PROJECT_AUDIT.md        # Audit trạng thái code hiện tại
project/TECHNICAL_COMMANDS.md   # Lệnh chạy preprocessing/training/evaluation/inference
project/PIPELINE_ROADMAP.md     # Sườn pipeline và roadmap phát triển đề xuất
```

## 3. Sườn Pipeline Tổng Quan

```text
Input scene image
        │
        ▼
Prepare 2D scene-level
        │  detect / segment object, tạo bbox, mask, label hoặc unknown
        ▼
Prepare object-level
        │  crop, remove background, resize/letterbox, normalize, augment
        ▼
Object reconstruction model
        │  pretrained encoder → latent vector → MLP decoder → point cloud
        ▼
Adaptation / semi-supervised learning
        │  PEFT, ADA, teacher-student, pseudo-label filtering nếu cần
        ▼
Output object set
        │  object_id, label/unknown, bbox, mask, point cloud/mesh, confidence
```

## 4. Vai Trò Các Kỹ Thuật Được Chọn

| Kỹ thuật | Vị trí trong pipeline | Vai trò |
| --- | --- | --- |
| Pix3D | Dataset object-level | Train/fine-tune reconstruction cho `chair`, `sofa`, `table`. |
| Pretrained encoder | Model reconstruction | Tận dụng feature ảnh từ ResNet/ViT/DINO/CLIP thay vì train encoder từ đầu. |
| Data augmentation | Object-level prepare | Giảm overfit feature bằng crop, blur, noise, brightness, contrast, affine. |
| Detection/segmentation | 2D scene-level prepare | Tách object khỏi scene lớn trước khi reconstruct. |
| PEFT | Encoder fine-tuning | Giảm tham số trainable bằng freeze, adapter, LoRA hoặc partial fine-tuning. |
| ADA | Latent feature sau encoder | Giảm domain shift giữa source có nhãn và target thực tế chưa có GT 3D. |
| Teacher-student | Semi-supervised training | Tận dụng target data không có GT 3D bằng consistency learning. |
| Pseudo-labeling | Semi-supervised training | Sinh pseudo 3D cho target, chỉ dùng sau khi lọc uncertainty/confidence. |

## 5. Workflow Code Hiện Tại

1. **Chuẩn bị dữ liệu Pix3D**: đặt dữ liệu raw tại `project/data/raw/pix3d`, gồm `pix3d.json`, ảnh, mask và CAD model.
2. **Làm sạch metadata**: `project/src/preprocessing/metadata_cleaner.py` kiểm tra cột bắt buộc, file tồn tại, category và sinh đường dẫn processed.
3. **Tạo dữ liệu processed**: `project/src/preprocessing/build_processed_dataset.py` crop bbox, apply mask, resize ảnh/mask; `mesh_processor.py` sample mesh `.obj` thành point cloud `.npy`.
4. **Train baseline**: `project/src/training/training_pipeline.py` dùng ResNet encoder + point cloud decoder dự đoán point cloud từ ảnh, tối ưu bằng Chamfer Distance.
5. **Evaluate**: `project/src/evaluation/evaluate_baseline.py` tính Chamfer Distance, F-score, precision, recall trên split train/val/test.
6. **Inference một ảnh**: `project/src/inference/baseline_inference.py` load checkpoint, xuất `.npy`, `.ply`, `.png`, JSON summary.
7. **API backend**: `server/main.py` expose `/health`, các endpoint mock scan, và `/reconstruct-image` để gọi baseline inference nếu checkpoint tồn tại.

## 6. Lệnh Chạy Nhanh

### Python pipeline

```powershell
cd project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

$env:KMP_DUPLICATE_LIB_OK="TRUE"
python main_workflow.py --categories chair --max-samples 256 --epochs 5 --batch-size 4 --overwrite
```

### Evaluate checkpoint

```powershell
cd project
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python -m src.evaluation.evaluate_baseline --split test --categories chair --batch-size 4 --output-dir results/baseline
```

### Inference một ảnh

```powershell
cd project
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python -m src.inference.baseline_inference --image data/processed/images/chair/2003.png --output-dir results/baseline/outputs/inference
```

### Backend

```powershell
cd server
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Kiểm tra:

```powershell
curl http://localhost:8000/health
curl -X POST "http://localhost:8000/reconstruct-image" -F "image=@..\project\data\processed\images\chair\2003.png"
```

## 7. Roadmap Triển Khai

```text
Phase 1: Baseline object reconstruction
Pix3D/source → preprocessing → model → point cloud → Chamfer Distance

Phase 2: Strong augmentation
Tăng blur, noise, crop, brightness, contrast để giảm overfit feature.

Phase 3: Scene-level prepare
Thêm detection/segmentation để tách object từ ảnh scene.

Phase 4: Pretrained encoder + PEFT
Thay hoặc bổ sung encoder pretrained, freeze phần lớn encoder, train decoder/adapter.

Phase 5: ADA
Source có GT 3D + target real crop chưa GT → domain-invariant latent feature.

Phase 6: Teacher-student / pseudo-labeling
Tận dụng target unlabeled bằng consistency hoặc pseudo-label đã lọc uncertainty.

Phase 7: Final inference system
Scene image → object set 2D → object set 3D → export `.ply`/`.obj`/metadata.
```

## 8. Các Điểm Chưa Khớp Cần Lưu Ý

- Pipeline ML hiện tại vẫn là **single image → point cloud**, chưa phải video 360 hoặc full scene mesh.
- Detection/video/status endpoint trong `server` hiện vẫn mock.
- Frontend source chưa đầy đủ so với cấu hình npm/Jest cũ.
- `tester/` đang kiểm thử CRUD blog, chưa phải test domain 3D reconstruction.
- Ảnh/mask/point cloud processed có thể chưa được commit; cần chạy preprocessing trước khi train/inference.

## 9. Kết Luận Thiết Kế

Sườn phù hợp nhất cho dự án là:

```text
Detection/segmentation để tách object
→ object-level preprocessing
→ Pix3D + pretrained encoder để học reconstruction
→ MLP decoder sinh point cloud
→ strong augmentation để giảm overfit
→ PEFT để giảm chi phí train
→ ADA để giảm domain shift
→ teacher-student hoặc pseudo-labeling để tận dụng target unlabeled data
```

Pix3D và pretrained model không bị thay thế bởi scene-level prepare. Pix3D là lõi dữ liệu cho object reconstruction, pretrained encoder là nền feature, còn scene-level prepare là cầu nối để áp dụng model lên ảnh thực tế có nhiều vật thể.
