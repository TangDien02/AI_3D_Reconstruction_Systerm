# 3DRecon FastAPI Server

Folder `server/` chua backend FastAPI cho workflow mobile -> YOLO -> TripoSR.

- **Object detection/selection**: `/detect-frame` va `/reconstruct-object` dung YOLO segmentation weights trong `server/weights/yolo26n-seg.pt`.
- **Core reconstruction**: `/reconstruct-object` crop bbox vat the, gui crop do sang TripoSR de tu remove background, dung mesh, sample point cloud, va export artifact.
- **Direct image reconstruction**: `/reconstruct-image` gui anh truc tiep vao TripoSR, phu hop de test core khong can frontend.

## Files

| File | Nhiem vu |
| --- | --- |
| `main.py` | FastAPI app, YOLO detect, TripoSR reconstruct, static result mounts. |
| `requirements.txt` | Backend dependencies gom FastAPI, YOLO, TripoSR runtime, rembg/onnxruntime. |
| `Dockerfile` | Dong goi backend. Can bo sung TripoSR repo/weights neu chay Docker. |

## Endpoints

| Method | Path | Mo ta |
| --- | --- | --- |
| `GET` | `/health` | Tra detector config, TripoSR config, va trang thai core da load hay chua. |
| `POST` | `/detect-frame` | Nhan frame camera, tra bbox YOLO de frontend cho nguoi dung chon object. |
| `POST` | `/segment-object` | Luu artifact crop/mask/debug, gom `triposr_crop`. |
| `POST` | `/reconstruct-object` | YOLO chon object, tao `triposr_crop`, goi TripoSR, tra GLB/PLY/preview. |
| `POST` | `/reconstruct-image` | Gui mot anh truc tiep vao TripoSR, tra GLB/PLY/preview. |
| `POST` | `/upload-scan-video` | Mock video workflow, chua noi core nhieu frame. |
| `GET` | `/scan-status/{job_id}` | Mock status cho video workflow. |

## TripoSR Setup

TripoSR code duoc import tu repo chinh thuc `VAST-AI-Research/TripoSR`. Clone repo do vao `external/TripoSR` hoac cau hinh bien moi truong `TRIPOSR_REPO_DIR`.

```powershell
cd ..
git clone https://github.com/VAST-AI-Research/TripoSR.git external/TripoSR
cd server
pip install -r requirements.txt
```

Bien moi truong hay dung:

```powershell
$env:TRIPOSR_REPO_DIR="C:\Users\pminh\Desktop\MyProject\AI_3D_Reconstruction_Systerm_TangDien02\external\TripoSR"
$env:TRIPOSR_MODEL_SAVE_FORMAT="glb"
$env:TRIPOSR_MC_RESOLUTION="256"
$env:TRIPOSR_NUM_POINTS="2048"
$env:TRIPOSR_REMOVE_BACKGROUND="true"
```

## Run Server

```powershell
cd server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Check:

```powershell
curl http://localhost:8000/health
curl -X POST "http://localhost:8000/reconstruct-image" -F "image=@..\project\results\triposr_core_test\chair_smoke\input.png"
```

Khi thanh cong, response co:

- `files.mesh_glb`: mesh chinh de xem GLB.
- `files.mesh_colored_ply`: mesh PLY co vertex color de check mau trong Blender.
- `files.pointcloud_ply`: point cloud sampled tu mesh.
- `files.triposr_input`: anh sau buoc remove background/resize foreground.
- `files.preview_png`: preview point cloud.
