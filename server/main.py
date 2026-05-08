from fastapi import FastAPI, File, Form, UploadFile

app = FastAPI(title="3DRecon API")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/detect-frame")
async def detect_frame(image: UploadFile = File(...)):
    return {
        "objects": [
            {
                "id": "mock_object_1",
                "label": "object",
                "confidence": 0.92,
                "bbox": {"x": 120, "y": 180, "width": 240, "height": 300},
            }
        ]
    }


@app.post("/upload-scan-video")
async def upload_scan_video(
    video: UploadFile = File(...),
    selected_object: str | None = Form(default=None),
):
    return {
        "job_id": "mock_scan_001",
        "status": "queued",
        "filename": video.filename,
        "selected_object": selected_object,
    }


@app.get("/scan-status/{job_id}")
def get_scan_status(job_id: str):
    return {
        "job_id": job_id,
        "status": "done",
        "progress": 100,
        "model_url": "http://localhost:8000/models/mock_scan_001.glb",
    }
