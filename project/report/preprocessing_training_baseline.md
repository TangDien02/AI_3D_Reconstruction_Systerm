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
project/src/models/transformer_pointcloud.py
```

Kien truc baseline:

1. Anh dau vao `[3, 224, 224]`.
2. Patch embedding bang Conv2D.
3. Transformer encoder hoc dac trung anh.
4. Decoder MLP du doan point cloud.
5. Loss chinh: Chamfer Distance.
6. Metric danh gia: Chamfer Distance va F-score.

Training pipeline nam trong:

```text
project/src/training/training_pipeline.py
```

Lenh training baseline da chay:

```bash
cd project
set KMP_DUPLICATE_LIB_OK=TRUE
python -m src.training.training_pipeline --dataset-mode processed --categories chair --max-samples 256 --epochs 5 --batch-size 4
```

Ghi chu: `KMP_DUPLICATE_LIB_OK=TRUE` duoc dung tam thoi do moi truong Anaconda tren Windows gap xung dot OpenMP. Ve lau dai nen tao virtual environment sach cho PyTorch.

## 6. Ket qua baseline

Ket qua duoc luu tai:

```text
project/results/training/metrics/training_metrics.csv
project/results/training/checkpoints/transformer_pointcloud_net.pt
```

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

## 7. Trang thai hien tai va viec nen lam tiep

Da hoan thanh:

- Metadata sach.
- Split train/val/test.
- Anh va mask da xu ly.
- Point cloud `.npy`.
- Processed dataloader.
- Transformer point cloud baseline.
- Training baseline va checkpoint.

Can cai thien tiep:

- Tao moi truong Python sach de bo workaround `KMP_DUPLICATE_LIB_OK`.
- Chay baseline tren nhieu category hon, vi hien tai moi train `chair`.
- Luu them bieu do loss/metric theo epoch.
- Them script evaluate rieng tren `test.csv`.
- Ket noi checkpoint voi backend inference neu can demo end-to-end.
