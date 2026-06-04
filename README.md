# AI 3D Reconstruction System

Repo nay da duoc rut gon cho luong runtime hien tai:

```text
Expo mobile app -> FastAPI backend -> Hunyuan3D worker -> mesh / texture export
```

TripoSR da bi go khoi project. Backend chi ho tro reconstruction qua
`RECONSTRUCTION_BACKEND=hunyuan_remote`.

## Runtime duoc ho tro

- Python: `3.11.x`
- Node.js: `20.x`

## Thu muc chinh

- `mobile/`: Expo app chup camera, gui anh/bbox va hien thi GLB/texture.
- `server/`: FastAPI backend, artifact API, Hunyuan remote client.
- `scripts/`: setup, Google Cloud VM worker/backend helpers.
- `notebook/`: Hunyuan VM/Cloudflare/Jupyter workflows.
- `project/samples/`: input sample cho smoke test.

## Env local

Khong ghi secret vao `.env` vi file nay dang bi Git track tu lich su cu.
Dung `.env.local` cho cau hinh local:

```powershell
RECONSTRUCTION_BACKEND=hunyuan_remote
HUNYUAN_REMOTE_URL=https://<your-worker-tunnel>
IMAGE_CLEANER_BACKEND=auto
ENABLE_REMBG_CLEANER=true
CLEAN_IMAGE_MAX_SIDE=1536
CLEAN_IMAGE_PAD_RATIO=0.08
```

`.env.local` da nam trong `.gitignore`.

Gemini/Nano Banana API da bi go khoi runtime. Local cleaner se thu `rembg`
neu co cai, roi fallback ve `crop_only` neu `rembg` thieu hoac fail:

```powershell
pip install rembg onnxruntime
```

## Setup tren Windows

```powershell
cd C:\Users\pminh\Desktop\MyProject\AI_3D_Reconstruction_Systerm_TangDien02
.\scripts\setup.ps1
```

Script se:

1. Tim `Python 3.11`
2. Tao lai `.venv`
3. Cai `requirements.txt`
4. Chay import check runtime

## Chay backend voi Hunyuan worker

```powershell
cd C:\Users\pminh\Desktop\MyProject\AI_3D_Reconstruction_Systerm_TangDien02
.\.venv\Scripts\Activate.ps1
$env:RECONSTRUCTION_BACKEND="hunyuan_remote"
$env:HUNYUAN_REMOTE_URL="https://<your-worker-tunnel>"
$env:HUNYUAN_REMOTE_OUTPUT_FORMAT="glb"
$env:HUNYUAN_REMOTE_ENABLE_TEXTURE="false"
$env:HUNYUAN_REMOTE_TIMEOUT_SECONDS="1800"
$env:HUNYUAN_REMOTE_POLL_INTERVAL_SECONDS="5"
python -m uvicorn server.main:app --host 0.0.0.0 --port 8000
```

Backend giu nguyen API cho Expo, nhung reconstruct se goi Hunyuan worker:

- `POST /reconstruct-bbox` nhan anh + bbox nguoi dung keo, crop, local clean, roi goi Hunyuan.
- `POST /preprocess/clean-image` debug crop + local clean, khong goi Hunyuan.
- `GET /reconstruction-jobs/<job_id>` de poll trang thai `cropping`, `cleaning`, `generating_shape`, `completed`, `failed`.
- `POST <HUNYUAN_REMOTE_URL>/start-shape` cho shape.
- `POST <HUNYUAN_REMOTE_URL>/start-texture` cho texture paint rieng.

Khuyen nghi production/T4:

```powershell
$env:HUNYUAN_REMOTE_ENABLE_TEXTURE="false"
```

Sau khi co shape `job_id`, paint texture bang:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/paint-texture" -F "job_id=<JOB_ID>"
```

## Smoke test reconstruct-image

```powershell
curl.exe -X POST "http://127.0.0.1:8000/reconstruct-image" `
  -F "image=@project\samples\chair_demo.png"
```

Artifact duoc ghi vao:

- `server/models/<job_id>/mesh.glb`
- `server/models/<job_id>/mesh_textured.glb` neu paint texture
- `server/models/<job_id>/reconstruction_summary.json`

## Chay mobile

```powershell
cd C:\Users\pminh\Desktop\MyProject\AI_3D_Reconstruction_Systerm_TangDien02\mobile
npm install
$env:EXPO_PUBLIC_API_BASE_URL="http://<your-lan-ip>:8000"
npm start
```

Android Emulator:

```powershell
$env:EXPO_PUBLIC_API_BASE_URL="http://10.0.2.2:8000"
```

## Docker

```powershell
docker compose up --build backend
```

Set `HUNYUAN_REMOTE_URL` trong shell truoc khi chay Docker Compose.

## Google Cloud

Huong dan deploy/chay tu mot clone moi nam trong:

```text
GOOGLE_CLOUD_DEPLOY.md
```

Bo huong dan final de dung lai VM/backend khi instance bi kill nam trong:

```text
deploy/
```

Trong do co `FINAL_VM_DEPLOY.ipynb`, `APP_FEATURES_AND_USAGE.md`,
`requirements.txt`, `COMMANDS.md`, va cac script tmux/backend helper.

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
