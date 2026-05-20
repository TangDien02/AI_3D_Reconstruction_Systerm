# 3D Reconstruction Pipeline Roadmap

Tài liệu này mô tả sườn chạy hệ thống từ input đến output, dựa trên định hướng đã chọn: **Pix3D/pretrained model cho object reconstruction**, **2D scene-level prepare để tách object**, và các kỹ thuật mở rộng như **PEFT**, **ADA**, **teacher-student**, **pseudo-labeling** để xử lý GPU, domain shift và dữ liệu chưa nhãn.

## 1. Mục Tiêu Hệ Thống

```text
Input:
Ảnh scene / ảnh phòng / ảnh thực tế có một hoặc nhiều vật thể.

Output:
Tập vật thể 3D riêng lẻ, có thể có nhãn hoặc unknown.
```

Output mục tiêu:

```json
{
  "scene_id": "scene_001",
  "objects": [
    {
      "object_id": "obj_001",
      "label": "chair",
      "label_status": "predicted",
      "confidence": 0.92,
      "bbox": [120, 80, 260, 310],
      "mask_path": "masks/scene_001_obj_001.png",
      "pointcloud_path": "outputs/scene_001_obj_001.ply",
      "domain": "real"
    }
  ]
}
```

## 2. Pipeline Tổng Thể

```text
Scene image
  → Prepare 2D scene-level
  → Prepare object-level
  → Object reconstruction model
  → Training/adaptation strategy
  → Evaluation
  → Object set output
```

Sơ đồ chi tiết:

```text
INPUT SCENE IMAGE
        │
        ▼
1. 2D SCENE-LEVEL PREPARE
        │
        ├─ image preprocessing
        ├─ object detection / instance segmentation
        ├─ object filtering
        └─ crop + mask + metadata
        │
        ▼
2. OBJECT-LEVEL PREPARE
        │
        ├─ background removal
        ├─ padding + resize / letterbox
        ├─ normalization
        ├─ train-time augmentation
        └─ object tensor
        │
        ▼
3. OBJECT RECONSTRUCTION MODEL
        │
        ├─ pretrained encoder
        ├─ optional PEFT / partial fine-tuning
        ├─ latent vector
        └─ MLP decoder → point cloud
        │
        ▼
4. ADAPTATION / SEMI-SUPERVISED LEARNING
        │
        ├─ strong augmentation
        ├─ ADA at latent feature
        ├─ teacher-student consistency
        └─ pseudo-labeling + uncertainty filtering
        │
        ▼
OUTPUT OBJECT SET
```

## 3. Prepare 2D Scene-Level

Mục tiêu: từ ảnh/cảnh lớn, phát hiện và tách từng vật thể.

```text
Input scene image
→ image preprocessing
→ detection / instance segmentation
→ filter object
→ crop + mask từng object
→ scene-level metadata
```

Nhiệm vụ:

| Bước | Mô tả | Output |
| --- | --- | --- |
| Image preprocessing | Resize, chuẩn hóa màu/ánh sáng, denoise nhẹ nếu cần. | ảnh scene chuẩn |
| Object localization | Tìm vùng chứa object bằng detector. | bbox, class, confidence |
| Instance segmentation | Tạo mask từng object nếu có segmenter. | mask object |
| Object filtering | Loại object quá nhỏ, quá mờ, confidence thấp. | object hợp lệ |
| Crop generation | Crop theo bbox/mask, thêm padding. | object crop |
| Metadata | Lưu `scene_id`, `object_id`, `bbox`, `mask_path`, `crop_path`, `label`. | record object |

Nếu object chưa có nhãn chắc chắn, dùng:

```text
label = "unknown"
label_status = "unknown" hoặc "predicted"
```

## 4. Prepare Object-Level

Mục tiêu: chuẩn hóa từng object crop trước khi đưa vào model reconstruction.

```text
object crop + mask
→ remove background
→ resize / letterbox
→ normalize
→ augmentation nếu train
→ image tensor
```

Nhiệm vụ:

| Bước | Mô tả |
| --- | --- |
| Background removal | Dùng mask để giữ object, giảm background bias. |
| Crop padding | Thêm 5-15% padding để tránh cắt mất chân/cạnh object. |
| Resize / letterbox | Giữ aspect ratio, đưa về `224x224` hoặc `256x256`. |
| Normalize | Dùng ImageNet mean/std nếu encoder pretrained từ ImageNet. |
| Train augmentation | Dùng transform mạnh hơn để giảm overfit feature. |
| Val/test transform | Chỉ resize/letterbox/normalize, không random augmentation. |

Augmentation nên thử:

```text
RandomResizedCrop
RandomHorizontalFlip
ColorJitter brightness / contrast / saturation
GaussianBlur
Add noise custom transform
RandomAffine hoặc RandomRotation nhẹ
RandomErasing nếu không phá hình dạng chính
```

## 5. Dataset Strategy

Nguồn dữ liệu chính:

| Dataset/domain | Vai trò |
| --- | --- |
| Pix3D | Object-level image + mask/bbox + 3D CAD, phù hợp train/fine-tune `chair`, `sofa`, `table`. |
| ShapeNet / synthetic render | Source domain có GT 3D rõ, dùng mở rộng dữ liệu. |
| Real scene crops | Target domain thực tế, thường không có GT 3D, dùng cho ADA/consistency/pseudo-labeling. |

Nguyên tắc chống leakage:

```text
Split theo object_id / model_id, không split theo image view.
Augmentation chỉ thực hiện sau khi split.
Không để cùng một CAD model/object xuất hiện ở cả train và test.
Fit normalization/scaler chỉ trên train nếu có bước thống kê dữ liệu.
```

## 6. Object Reconstruction Model

Baseline mục tiêu:

```text
object image
→ pretrained encoder
→ latent vector
→ MLP decoder
→ point cloud N x 3
```

Khuyến nghị triển khai:

```text
Stage 1:
ResNet18/ResNet50 hoặc model hiện có làm baseline.

Stage 2:
Bỏ classification head, chỉ lấy feature encoder.

Stage 3:
MLP decoder sinh point cloud.

Stage 4:
Train bằng Chamfer Distance.
```

Point cloud được ưu tiên trước vì:

```text
Dễ triển khai hơn mesh/voxel.
Không cần thứ tự điểm cố định.
Phù hợp Chamfer Distance và F-score.
Có thể chuyển tiếp sang mesh bằng Poisson/Ball Pivoting ở phase sau.
```

## 7. PEFT / Partial Fine-Tuning

PEFT nằm trong phần encoder fine-tuning, không nằm ở data pipeline.

```text
object image
→ pretrained encoder + PEFT
→ latent vector
→ MLP decoder
→ point cloud
```

Cách dùng:

```text
ResNet:
Freeze early layers, fine-tune layer cuối hoặc thêm adapter nhỏ.

ViT/DINO/CLIP:
Dùng LoRA/Adapter trên attention/projection layers.

Decoder:
Train đầy đủ MLP decoder.
```

Mục tiêu:

```text
Giảm trainable parameters.
Giảm optimizer state và VRAM so với full fine-tuning.
Tận dụng pretrained feature khi dataset nhỏ.
```

## 8. ADA - Adversarial Domain Adaptation

ADA dùng khi có domain shift:

```text
Source: Pix3D / synthetic có GT 3D.
Target: real object crops không có GT 3D.
```

Vị trí:

```text
encoder → latent vector → Gradient Reversal Layer → domain discriminator
```

Pipeline:

```text
Source crop ─┐
             ├→ encoder → latent → decoder → point cloud
Target crop ─┘              │
                            ▼
                 Gradient Reversal Layer
                            ▼
                 Domain Discriminator
                            ▼
                    domain classification loss
```

Loss:

```text
L_total = L_chamfer_source + lambda_domain * L_domain
```

Vai trò:

```text
Làm latent feature bớt phụ thuộc domain.
Giúp decoder học từ source hoạt động ổn hơn trên target real crop.
```

## 9. Teacher-Student Và Pseudo-Labeling

Hai kỹ thuật này nằm ở semi-supervised training cho target data chưa có GT 3D.

### Teacher-Student Consistency

```text
Target weak augmentation → Teacher → prediction ổn định
Target strong augmentation → Student → prediction
Teacher vs Student → consistency loss
```

Loss có thể dùng:

```text
L_consistency = Chamfer(student_pointcloud, teacher_pointcloud)
```

Hoặc nhẹ hơn:

```text
L_latent = MSE(student_latent, teacher_latent)
```

### Pseudo-Labeling

Pseudo-labeling không nên dùng ngay từ đầu. Thứ tự an toàn:

```text
Train baseline bằng labeled source.
Thêm strong augmentation và/hoặc ADA để giảm overfit/domain gap.
Dùng teacher/baseline sinh pseudo point cloud cho target crop.
Lọc pseudo-label đáng tin.
Train lại bằng labeled + pseudo-labeled data.
```

Filtering nên dựa trên:

```text
Detector/mask confidence cao.
Prediction ổn định qua nhiều augmentation.
Uncertainty thấp.
Không bị che khuất nặng.
Shape không dị thường theo category prior nếu có nhãn.
```

Loss mở rộng:

```text
L_total =
  L_chamfer_source
  + lambda_pseudo * L_chamfer_pseudo
  + lambda_consistency * L_consistency
  + lambda_domain * L_domain
```

## 10. Training Roadmap

Không nên bật tất cả kỹ thuật ngay từ đầu. Nên triển khai theo phase:

```text
Phase 1: Baseline object reconstruction
Pix3D/source → preprocessing → model → point cloud → Chamfer Distance.

Phase 2: Strong augmentation
Tăng blur, noise, crop, brightness, contrast để giảm overfit feature.

Phase 3: Scene-level prepare
Thêm detector/segmenter để tách object từ ảnh scene.

Phase 4: Pretrained encoder + PEFT
Tận dụng ResNet/ViT pretrained, freeze encoder, train decoder/adapter.

Phase 5: ADA
Source + target real crop → domain-invariant latent feature.

Phase 6: Teacher-student / pseudo-labeling
Tận dụng target unlabeled bằng consistency hoặc pseudo-label đã lọc.

Phase 7: Final inference
Scene image → object set 2D → object set 3D → export `.ply`/`.obj` + metadata.
```

## 11. Evaluation

Metric reconstruction:

```text
Chamfer Distance
F-score
Precision
Recall
```

Metric theo pipeline:

```text
Detection/segmentation confidence
Object crop quality
Domain accuracy nếu dùng ADA
Uncertainty score nếu dùng teacher-student/pseudo-labeling
Per-category metric cho chair / sofa / table
```

Checkpoint nên lưu:

```text
best_model.pt theo val_chamfer_distance
last_checkpoint.pt
training_metrics.csv
inference metadata JSON
```

## 12. Mapping Với Code Hiện Tại

| Roadmap | Trạng thái hiện tại |
| --- | --- |
| Pix3D metadata cleaning | Có trong `src/preprocessing/metadata_cleaner.py`. |
| Object crop/mask/resize | Có trong `src/preprocessing/build_processed_dataset.py`. |
| Mesh to point cloud | Có trong `src/preprocessing/mesh_processor.py`. |
| Single-object training | Có trong `src/training/training_pipeline.py`. |
| Chamfer/F-score | Có trong `src/metrics/losses.py`. |
| Inference single image | Có trong `src/inference/baseline_inference.py`. |
| Scene-level detection/segmentation | Mới có fallback/mocking, cần phát triển thật. |
| Pretrained ResNet/ViT + PEFT | Chưa tích hợp vào training chính. |
| ADA | Chưa tích hợp. |
| Teacher-student/pseudo-labeling | Chưa tích hợp. |
| Mesh/GLB output | Chưa có, hiện output chính là point cloud. |

## 13. Kết Luận

Pix3D và pretrained model là lõi học object reconstruction. Scene-level prepare không thay thế Pix3D, mà là cầu nối để dùng model trên ảnh thực tế nhiều vật thể.

Sườn thiết kế ổn định nhất:

```text
2D scene prepare
→ object-level prepare
→ pretrained encoder + MLP decoder
→ Chamfer-supervised source training
→ strong augmentation
→ PEFT nếu cần giảm chi phí
→ ADA nếu có source/target domain shift
→ teacher-student hoặc pseudo-labeling nếu có target unlabeled
→ object set 3D output
```
