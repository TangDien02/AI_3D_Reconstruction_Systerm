# 3DRecon FastAPI Server

Folder `server/` chứa backend FastAPI cho hệ thống 3D reconstruction. Hiện server có hai nhóm chức năng:

- **Scan workflow mock** để giữ contract API cho client tương lai.
- **Image reconstruction thật ở mức baseline** qua checkpoint trong `project/results/baseline`.

## File Và Nhiệm Vụ

| File | Nhiệm vụ |
| --- | --- |
| `main.py` | Khởi tạo FastAPI app, định nghĩa endpoint health, mock detection/video scan, và endpoint inference ảnh đơn. |
| `vit_reconstruction.py` | Mô hình ViT minh họa gồm patch embedding, Transformer block, training setup và hàm xử lý frame thành feature. Chưa phải model checkpoint chính. |
| `requirements.txt` | Dependency backend. |
| `Dockerfile` | Đóng gói backend để chạy qua Docker/Docker Compose. |

## Endpoint

| Method | Path | Trạng thái | Mô tả |
| --- | --- | --- | --- |
| `GET` | `/health` | Thật | Trả `status` và kiểm tra checkpoint baseline có tồn tại không. |
| `POST` | `/detect-frame` | Mock | Nhận ảnh, trả một bounding box giả lập. Chưa chạy YOLO thật. |
| `POST` | `/upload-scan-video` | Mock | Nhận video, trả `job_id` giả lập. Chưa enqueue job hoặc lưu pipeline xử lý video. |
| `GET` | `/scan-status/{job_id}` | Mock | Trả trạng thái `done` và URL model giả lập. |
| `POST` | `/reconstruct-image` | Baseline thật | Lưu ảnh upload, gọi `project/src/inference/baseline_inference.py`, xuất `.npy`, `.ply`, `.png`. |

## Điều Kiện Để Chạy `/reconstruct-image`

Endpoint này cần checkpoint:

```text
project/results/baseline/outputs/checkpoints/transformer_pointcloud_net.pt
```

Nếu checkpoint chưa có, train baseline từ folder `project/`:

```powershell
cd ..\project
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python main_workflow.py --skip-preprocessing --categories chair --epochs 5 --batch-size 4
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

- Thay mock `/detect-frame` bằng YOLO/segmentation thật hoặc gọi `ImagePreprocessor`.
- Thay mock video scan bằng job queue và pipeline xử lý nhiều frame.
- Đồng bộ `vit_reconstruction.py` với model thật `TransformerPointCloudNet`, hoặc ghi rõ đây chỉ là prototype.
- Nếu dùng Docker Compose với Floci/AWS env, cần thêm code lưu upload/result vào storage thay vì chỉ lưu local.
