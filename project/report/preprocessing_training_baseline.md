# Preprocessing And Training Baseline Report

## 1. Muc tieu

Bao cao nay ghi lai pipeline tien xu ly du lieu Pix3D va baseline training cho bai toan single-view 3D reconstruction. Muc tieu hien tai la bien du lieu tho gom anh RGB, mask va CAD model thanh dataset da xu ly, sau do train mot baseline du doan point cloud 3D tu anh 2D.

## 2. Cau truc du lieu da xu ly

Thu muc du lieu chinh:

```text
project/data/processed/
  pix3d_clean_metadata.csv
  images/
  masks/
  points/
  splits/
    train.csv
    val.csv
    test.csv
```

Thong ke du lieu sau tien xu ly:

| Thanh phan | So luong |
| --- | ---: |
| Clean metadata samples | 10,069 |
| Processed RGB images | 10,069 |
| Processed masks | 10,069 |
| Processed point clouds | 395 |
| Train split rows | 7,043 |
| Validation split rows | 1,508 |
| Test split rows | 1,518 |

Phan bo category:

| Category | So mau |
| --- | ---: |
| chair | 3,839 |
| sofa | 1,947 |
| table | 1,870 |
| bed | 994 |
| desk | 700 |
| bookcase | 361 |
| wardrobe | 243 |
| misc | 68 |
| tool | 47 |

## 3. Pipeline tien xu ly

Pipeline tien xu ly nam trong:

```text
project/src/preprocessing/
  metadata_cleaner.py
  build_processed_dataset.py
  mesh_processor.py
  image_processor.py
```

Luong xu ly:

1. Doc `pix3d.json` tu `data/raw/pix3d`.
2. Kiem tra file anh, mask va model ton tai.
3. Ghi metadata sach vao `data/processed/pix3d_clean_metadata.csv`.
4. Chia dataset thanh `train.csv`, `val.csv`, `test.csv`.
5. Crop anh theo bounding box, apply mask, resize ve `224x224`.
6. Luu anh da xu ly vao `data/processed/images`.
7. Luu mask da xu ly vao `data/processed/masks`.
8. Sample CAD mesh thanh point cloud, normalize ve unit sphere.
9. Luu point cloud `.npy` vao `data/processed/points`.

Lenh chay preprocessing day du:

```bash
cd project
python -m src.preprocessing.build_processed_dataset --progress-interval 100
```

Neu can chay rieng point cloud:

```bash
cd project
python -m src.preprocessing.mesh_processor --progress-interval 25
```

## 4. Dataset cho training

Dataset training chinh hien tai la `ProcessedPix3DDataset` trong:

```text
project/src/data/dataloader.py
```

Moi sample tra ve:

```python
{
    "image": image_tensor,        # [3, 224, 224]
    "category": category_name,
    "points_gt": point_tensor,    # [N, 3]
    "pointcloud_path": path,
    "image_path": path,
}
```

Dataset nay doc truc tiep du lieu da xu ly tu `data/processed`, nen training khong can crop anh hoac sample mesh lap lai moi epoch.

## 5. Baseline model

Model baseline nam trong:

```text
project/src/models/object_reconstruction.py
```

Kien truc baseline:

1. Anh dau vao `[3, 224, 224]`.
2. ResNet encoder hoc dac trung anh.
3. Decoder MLP du doan point cloud.
4. Loss chinh: Chamfer Distance.
5. Metric danh gia: Chamfer Distance va F-score.

Training pipeline nam trong:

```text
project/src/training/training_pipeline.py
```

Danh gia va inference baseline nam trong:

```text
project/src/evaluation/evaluate_baseline.py
project/src/inference/baseline_inference.py
project/src/utils/pointcloud_io.py
```

Lenh training baseline da chay:

```bash
cd project
set KMP_DUPLICATE_LIB_OK=TRUE
python -m src.training.training_pipeline --dataset-mode processed --categories chair --epochs 5 --batch-size 2 --output-dir results/chair_resnet_baseline
```

Ghi chu: `KMP_DUPLICATE_LIB_OK=TRUE` duoc dung tam thoi do moi truong Anaconda tren Windows gap xung dot OpenMP. Ve lau dai nen tao virtual environment sach cho PyTorch.

## 6. Ket qua baseline

Quy trinh baseline da duoc bo sung de ghi day du ket qua vao dung thu muc:

```text
project/results/chair_resnet_baseline/
  logs/
    baseline.log
  metrics/
    training_metrics.csv
  outputs/
    baseline_summary.json
    training_curves.png
    checkpoints/
      resnet_pointcloud_net.pt
```

Y nghia tung phan:

- `logs/baseline.log`: log qua trinh train, cau hinh dataset, so sample train/validation va metric theo epoch.
- `metrics/training_metrics.csv`: bang metric theo epoch dung de ve bieu do va so sanh cac lan chay.
- `outputs/checkpoints/resnet_pointcloud_net.pt`: checkpoint model baseline.
- `outputs/baseline_summary.json`: tom tat cau hinh chay baseline va duong dan artifact.
- `outputs/training_curves.png`: bieu do train loss, validation Chamfer Distance va validation F-score.

Bang ket qua training voi category `chair`, 256 samples, 5 epochs:

| Epoch | Train Loss | Val Chamfer Distance | Val F-score |
| ---: | ---: | ---: | ---: |
| 1 | 0.025991 | 0.017606 | 0.4138 |
| 2 | 0.017905 | 0.016433 | 0.4169 |
| 3 | 0.017214 | 0.017144 | 0.3993 |
| 4 | 0.017659 | 0.016693 | 0.4092 |
| 5 | 0.017368 | 0.016740 | 0.4079 |

Nhan xet ngan:

- Loss giam manh tu epoch 1 sang epoch 2.
- Val Chamfer Distance on dinh quanh `0.016-0.017`.
- Val F-score cao nhat o epoch 2 voi `0.4169`.
- Baseline da chay duoc end-to-end tu processed image sang predicted point cloud.

## 7. Benchmark hieu nang tuan 4

Muc tieu benchmark tuan 4 la danh gia cac cai tien training tren GPU CUDA:

- AMP mixed precision.
- ReduceLROnPlateau learning-rate scheduler.
- Phase freeze/unfreeze encoder: `--freeze-encoder --unfreeze-epoch 6`.
- Train augmentation duoc bat bang `--augment`.

Thiet lap chung cho cac benchmark:

```text
dataset_mode=processed
processed_dir=data/processed_2048
category=chair
encoder=resnet50
feature_dim=2048
num_points=2048
batch_size=16
max_samples=1024
val_max_samples=256
device=cuda
pretrained=True
skip_evaluation=True
skip_comparison=True
```

### 7.1 Benchmark 12 epoch

Ket qua 12 epoch cho thay AMP tang toc ro ret, dac biet sau khi unfreeze encoder.
`baseline.log` co 2 lan chay baseline A, nen dung range/average de nhin do on dinh.

| Run | AMP | Scheduler | Total time | Frozen avg | Unfrozen avg | Best CD | Final F-score |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| A run 1 | off | off | 269.7s | 17.20s | 24.82s | 0.011249 | 0.5816 |
| A run 2 | off | off | 264.1s | 17.17s | 24.90s | 0.011559 | 0.5662 |
| B | on | off | 207.1s | 15.42s | 17.99s | 0.012140 | 0.5583 |
| C | off | on | 267.6s | 16.81s | 25.17s | 0.011524 | 0.5713 |
| D | on | on | 208.8s | 15.42s | 18.13s | 0.011787 | 0.5740 |

Nhan xet:

- AMP giam tong thoi gian training khoang 22% so voi baseline khong AMP.
- Sau unfreeze encoder, AMP giam thoi gian moi epoch khoang 27%.
- ReduceLROnPlateau trong benchmark 12 epoch chua giam learning rate vi validation metric van tiep tuc cai thien.
- Best Chamfer Distance giua cac run co chenh lech nho va bi anh huong boi shuffle/augmentation, nen 12 epoch chua du de ket luan ve chat luong model.

### 7.2 Benchmark scheduler 30 epoch

Sau khi xac nhan AMP co loi ich ro, benchmark tiep theo giu AMP bat trong ca hai run va chi so sanh scheduler on/off.

| Run | AMP | Scheduler | Epoch chay | Total time | Best epoch | Best CD | Final F-score | LR reduction |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| E | on | off | 30 | 548.5s | 28 | 0.010498 | 0.6035 | none |
| F | on | on | 21 | 373.7s | 15 | 0.011007 | 0.5909 | epoch 17, 21 |

Chi tiet scheduler run F:

```text
epoch 17: decoder lr 1e-4 -> 5e-5, encoder lr 1e-5 -> 5e-6
epoch 21: decoder lr 5e-5 -> 2.5e-5, encoder lr 5e-6 -> 2.5e-6
early stopping at epoch 21
```

Nhan xet:

- Scheduler da hoat dong dung co che: phat hien plateau, giam learning rate va ket hop voi early stopping.
- Tuy nhien, voi cau hinh da benchmark `factor=0.5`, `patience=3`, `threshold=0.0001`, scheduler cho ket qua kem hon AMP-only:
  - Best CD cua E tot hon F: `0.010498` vs `0.011007`.
  - Final F-score cua E tot hon F: `0.6035` vs `0.5909`.
  - Run E tiep tuc cai thien den epoch 28, trong khi run F dung som o epoch 21.

### 7.3 Quyet dinh ky thuat

Ket luan tu benchmark tuan 4:

- Giu AMP vi day la cai tien hieu nang ro rang tren CUDA, giup giam thoi gian training ma khong lam hong pipeline.
- Giu code ReduceLROnPlateau trong pipeline de phuc vu experiment, nhung chua xem la cai tien mac dinh ve chat luong.
- Scheduler can duoc tuning them truoc khi dung mac dinh, vi cau hinh hien tai giam LR va dung som nhung chua cai thien Chamfer Distance/F-score.
- Sau benchmark, cau hinh scheduler duoc dieu chinh bot gat hon thanh `factor=0.7`, `patience=5` de giam LR cham hon va cho model them thoi gian cai thien.
- Khi bao cao, nen mo ta scheduler la "da tich hop va da benchmark, can danh gia them", con AMP la "duoc chap nhan giu lai".

## 8. Trang thai hien tai va viec nen lam tiep

Da hoan thanh:

- Metadata sach.
- Split train/val/test.
- Anh va mask da xu ly.
- Point cloud `.npy`.
- Processed dataloader.
- ResNet encoder + point cloud decoder baseline.
- Training baseline, log, metric, bieu do, summary va checkpoint trong `results/chair_resnet_baseline`.

Can cai thien tiep:

- Tao moi truong Python sach de bo workaround `KMP_DUPLICATE_LIB_OK`.
- Chay baseline tren nhieu category hon, vi hien tai moi train `chair`.
- Chay `evaluate_baseline.py` tren `test.csv` sau khi cai du `torch`.
- Mo rong backend inference tu anh don sang video/scan 360 neu can demo end-to-end.
- Chuyen giao dien sang dung endpoint `/reconstruct-image` sau khi backend ky thuat on dinh.

## 9. Lenh ky thuat rut gon

Danh sach lenh cai dat, preprocessing, training, evaluation, inference va backend da duoc gom tai:

```text
project/TECHNICAL_COMMANDS.md
```
