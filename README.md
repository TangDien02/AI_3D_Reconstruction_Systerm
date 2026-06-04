# AI 3D Reconstruction System

Repo nay da duoc rut gon de chi giu luong runtime:

`mobile app -> FastAPI backend -> YOLO segmentation -> reconstruction backend -> mesh / point cloud export`

## Runtime duoc ho tro

- Python: `3.11.x`
- Node.js: `20.x`

Khong ho tro `Python 3.12+` cho setup mac dinh cua repo nay. TripoSR van pin `transformers==4.35.0`, va stack do de vo hon tren 3.12/3.14 vi `tokenizers` co the bi buoc build native.

## Thu muc duoc giu

- `mobile/`: Expo app chup camera va goi backend.
- `server/`: FastAPI backend, YOLO detect, reconstruct API.
- `project/src/reconstruction/`: TripoSR runner.
- `project/src/utils/`: point cloud export va preview.
- `project/external/TripoSR/`: source TripoSR da duoc vendor vao repo chinh.
- `project/torchmcubes/`: fallback local cho `torchmcubes`.
- `project/samples/`: input sample cho smoke test.
- `scripts/setup.ps1`: script setup moi truong cho Windows.
- `scripts/verify_runtime.py`: check import runtime.

## Setup tren Windows

Buoc setup chuan:

```powershell
cd C:\Users\pminh\Desktop\MyProject\AI_3D_Reconstruction_Systerm_TangDien02
.\scripts\setup.ps1
```

Script se:

1. Tim `Python 3.11`
2. Tao lai `.venv`
3. Cai `requirements.txt`
4. Chay import check cho backend va TripoSR runner

Neu script bao khong tim thay `Python 3.11`, cai `Python 3.11.x` roi chay lai.

## Chay backend

```powershell
cd C:\Users\pminh\Desktop\MyProject\AI_3D_Reconstruction_Systerm_TangDien02
.\.venv\Scripts\Activate.ps1
$env:TRIPOSR_MODEL_SAVE_FORMAT="glb"
python -m uvicorn server.main:app --host 0.0.0.0 --port 8000
```

`TRIPOSR_REPO_DIR` khong can set trong setup mac dinh. Chi dung bien moi truong nay neu anh di doi thu muc `project/external/TripoSR`.

## Chay backend voi Hunyuan worker tren Colab

Neu Colab dang expose worker qua tunnel:

```powershell
cd C:\Users\pminh\Desktop\MyProject\AI_3D_Reconstruction_Systerm_TangDien02
.\.venv\Scripts\Activate.ps1
$env:RECONSTRUCTION_BACKEND="hunyuan_remote"
$env:HUNYUAN_REMOTE_URL="https://<your-colab-tunnel>"
$env:HUNYUAN_REMOTE_OUTPUT_FORMAT="glb"
$env:HUNYUAN_REMOTE_ENABLE_TEXTURE="false"
python -m uvicorn server.main:app --host 0.0.0.0 --port 8000
```

Backend se giu nguyen API cho Expo, nhung phan reconstruct se goi remote worker thay vi TripoSR local.

- `HUNYUAN_REMOTE_ENABLE_TEXTURE=false`: goi `POST /generate-shape`, xuat mesh trang.
- `HUNYUAN_REMOTE_ENABLE_TEXTURE=true`: goi `POST /generate-textured-shape`, chay shape truoc roi texture/paint sau.
- Khuyen nghi Colab Free: de `false` mac dinh, tao shape truoc, sau do paint texture bang endpoint rieng:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/paint-texture" -F "job_id=<JOB_ID>"
```

Tunnel ngrok van chi can expose port cua worker Colab. Backend local chi can doi `HUNYUAN_REMOTE_URL` sang URL ngrok hien tai.

### Colab worker Hunyuan3D-2 standard

Trong Colab Free T4, nen dung ban goc `tencent/Hunyuan3D-2` nhung chay tach shape va texture de giam peak RAM/VRAM:

```bash
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git
cd Hunyuan3D-2
pip install -r requirements.txt
pip install -e .
cd hy3dgen/texgen/custom_rasterizer && python3 setup.py install && cd ../../..
cd hy3dgen/texgen/differentiable_renderer && python3 setup.py install && cd ../../..
pip install fastapi uvicorn python-multipart pyngrok
```

Copy `scripts/colab_hunyuan_worker.py` vao Colab, roi chay:

```bash
export HUNYUAN_MODEL_ID="tencent/Hunyuan3D-2"
export HUNYUAN_MODEL_SUBFOLDER="hunyuan3d-dit-v2-0"
export HUNYUAN_TEXGEN_MODEL_ID="tencent/Hunyuan3D-2"
export HUNYUAN_KEEP_SHAPE_PIPELINE="0"
export HUNYUAN_KEEP_TEXTURE_PIPELINE="0"
export HUNYUAN_INFERENCE_STEPS="20"
export HUNYUAN_OCTREE_RESOLUTION="320"
export HUNYUAN_NUM_CHUNKS="12000"
nohup uvicorn colab_hunyuan_worker:app --host 0.0.0.0 --port 8010 > worker.log 2>&1 &
```

Tao ngrok tunnel cho port `8010`, sau do set `HUNYUAN_REMOTE_URL` cua backend local bang URL ngrok do:

```python
from pyngrok import ngrok

ngrok.set_auth_token("NGROK_AUTH_TOKEN_CUA_BAN")
public_url = ngrok.connect(8010, "http")
print(public_url)
```

Kiem tra health:

```powershell
curl http://127.0.0.1:8000/health
```

## Smoke test reconstruct-image

```powershell
curl -X POST "http://127.0.0.1:8000/reconstruct-image" `
  -F "image=@project\samples\chair_demo.png"
```

Artifact duoc ghi vao:

- `server/models/<job_id>/mesh.glb` hoac `mesh.obj`
- `server/models/<job_id>/pointcloud.ply`
- `server/models/<job_id>/mesh_colored.ply`
- `server/models/<job_id>/preview.png`
- `server/models/<job_id>/reconstruction_summary.json`

## Chay mobile

```powershell
cd C:\Users\pminh\Desktop\MyProject\AI_3D_Reconstruction_Systerm_TangDien02\mobile
npm install
$env:EXPO_PUBLIC_API_BASE_URL="http://<your-lan-ip>:8000"
npm start
```

Neu dung Android Emulator:

```powershell
$env:EXPO_PUBLIC_API_BASE_URL="http://10.0.2.2:8000"
```

Neu dung dien thoai that, set `EXPO_PUBLIC_API_BASE_URL` bang LAN IP cua may chay backend, vi du:

```powershell
$env:EXPO_PUBLIC_API_BASE_URL="http://192.168.1.10:8000"
```

## Docker

Docker la fallback cho backend. Build context da tro ve root de backend thay duoc ca `server/` va `project/`.

```powershell
docker compose up --build backend
```

## Google Cloud

Huong dan deploy/chay tu mot clone moi nam trong:

```text
GOOGLE_CLOUD_DEPLOY.md
```

Tom tat:

- Backend co Dockerfile Python 3.11 va `cloudbuild.yaml` de build image cho Cloud Run.
- Hunyuan3D-2 shape/texture nen chay tren Google Compute Engine GPU VM bang:

```bash
bash scripts/gcp_hunyuan_worker_bootstrap.sh
```

- Backend VM co the chay bang:

```bash
export HUNYUAN_REMOTE_URL="https://YOUR_WORKER_8010_URL"
bash scripts/gcp_backend_vm_bootstrap.sh
```

- Copy `.env.example` thanh `.env` hoac set cac bien moi truong tu shell/Cloud Run.
