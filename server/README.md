# 3DRecon Server

Folder này dành cho backend Python/FastAPI của hệ thống 3DRecon.

MVP backend sẽ có các API:

- `POST /detect-frame`: nhận 1 ảnh từ app, trả về danh sách bounding box YOLO.
- `POST /upload-scan-video`: nhận video quét 360 độ, trả về `job_id`.
- `GET /scan-status/{job_id}`: trả trạng thái xử lý và `model_url` khi hoàn tất.

Trong giai đoạn đầu, server có thể trả dữ liệu mock để app hoàn thiện luồng trước khi tích hợp YOLO/tracking/reconstruction thật.
