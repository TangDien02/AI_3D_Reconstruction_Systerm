# 3DRecon FastAPI Server

Folder `server/` chứa backend FastAPI cho hệ thống 3D reconstruction. Hiện server có ba nhóm chức năng:

- **Scan workflow mock** để giữ contract API cho client tương lai.
- **YOLO object detection/segmentation** để chọn bbox/mask object.
- **Image-to-mesh reconstruction** qua TripoSR. Baseline ResNet point cloud vẫn còn nhưng là legacy fallback.

## File Và Nhiệm Vụ

| File | Nhiệm vụ |
| --- | --- |
| `main.py` | Khởi tạo FastAPI app, định nghĩa endpoint health, mock detection/video scan, và endpoint inference ảnh đơn. |
| `requirements.txt` | Dependency backend. |
| `Dockerfile` | Đóng gói backend để chạy qua Docker/Docker Compose. |

## Endpoint

| Method | Path | Trạng thái | Mô tả |
| --- | --- | --- | --- |
| `GET` | `/health` | Thật | Trả `status`, trạng thái YOLO, TripoSR, baseline legacy. |
| `POST` | `/detect-frame` | Thật | Nhận ảnh, chạy YOLO, trả bbox/class/confidence. |
| `POST` | `/segment-object` | Thật | Nhận ảnh + optional bbox/object id, YOLO mask hoặc SAM2 refine mask, lưu crop/mask/model input. |
| `POST` | `/reconstruct-object` | Thật | YOLO chọn object, optional SAM2 refine mask, preprocess crop/mask, gọi TripoSR, xuất mesh GLB/OBJ. |
| `POST` | `/upload-scan-video` | Mock | Nhận video, trả `job_id` giả lập. Chưa enqueue job hoặc lưu pipeline xử lý video. |
| `GET` | `/scan-status/{job_id}` | Mock | Trả trạng thái `done` và URL model giả lập. |
| `POST` | `/reconstruct-image` | Thật | Lưu ảnh upload, preprocess square pad, gọi backend reconstruction hiện tại. |

## Backend Reconstruction

Mặc định server dùng:

```text
RECON_BACKEND=triposr
```

TripoSR không được vendor vào repo này. Cần checkout/cài riêng rồi trỏ biến môi trường:

```powershell
git clone https://github.com/VAST-AI-Research/TripoSR C:\models\TripoSR
cd C:\models\TripoSR
python -m pip install --upgrade setuptools
pip install -r requirements.txt
pip install git+https://github.com/tatsy/torchmcubes.git

$env:TRIPOSR_DIR="C:\models\TripoSR"
$env:RECON_BACKEND="triposr"
$env:TRIPOSR_MODEL_SAVE_FORMAT="glb"
```

## SAM2 Mask Refinement

Mặc định server dùng YOLO mask để không yêu cầu dependency nặng. Nếu muốn refine mask bằng SAM2 trước khi đưa vào TripoSR:

```powershell
git clone https://github.com/facebookresearch/sam2 C:\models\sam2
cd C:\models\sam2
pip install -e .

$env:SAM2_ENABLED="true"
$env:SAM2_MODEL_ID="facebook/sam2-hiera-large"
$env:SAM2_DEVICE="cuda"
```

Nếu muốn bắt buộc SAM2, không fallback về YOLO khi lỗi:

```powershell
$env:SAM2_REQUIRED="true"
```

Health check sẽ trả `sam2.enabled`, `sam2.available` và `reconstruction_preprocess.mask_refinement`.

Neu muon quay ve baseline point cloud cu:

```powershell
$env:RECON_BACKEND="legacy_pointcloud"
$env:RECON_BASELINE_CHECKPOINT="..\project\results\all_categories_resnet50_2048pts_30ep_aug\outputs\checkpoints\best_model.pt"
```

## Chạy Server

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

## Điểm Cần Tích Hợp Tiếp

- Thay mock video scan bằng job queue và pipeline xử lý nhiều frame.
- Thêm mesh cleanup/decimation/texture baking kiểm soát chất lượng.
- Nếu dùng Docker Compose với Floci/AWS env, cần thêm code lưu upload/result vào storage thay vì chỉ lưu local.
