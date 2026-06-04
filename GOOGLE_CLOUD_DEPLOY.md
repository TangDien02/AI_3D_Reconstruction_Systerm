# Google Cloud Deployment

This repo supports two Google Cloud paths:

- Backend API: FastAPI server, YOLO detection, artifact storage under `server/models`.
- Hunyuan worker: GPU VM service for Hunyuan3D-2 shape and optional texture.

Recommended production topology:

```text
Expo mobile app -> FastAPI backend -> GPU VM Hunyuan worker -> GLB output
```

The Hunyuan worker needs a GPU VM. Do not expect the full Hunyuan shape/texture
pipeline to run reliably on a small CPU-only Cloud Run service.

## 1. Clone

```bash
git clone https://github.com/TangDien02/AI_3D_Reconstruction_Systerm.git
cd AI_3D_Reconstruction_Systerm
```

## 2. GPU VM worker

Create a Google Compute Engine VM with:

- NVIDIA T4/L4/A100 or better
- Debian 12 or Ubuntu 22.04/24.04
- 100GB+ boot disk
- 30GB+ RAM for shape + texture, 50GB+ preferred
- NVIDIA driver and CUDA 12 toolkit available before texture builds

On the VM:

```bash
export REPO_URL="https://github.com/TangDien02/AI_3D_Reconstruction_Systerm.git"
bash scripts/gcp_hunyuan_worker_bootstrap.sh
curl http://127.0.0.1:8010/health
```

Expose the worker with Cloudflare Tunnel or your preferred HTTPS ingress:

```bash
cloudflared tunnel --url http://127.0.0.1:8010
```

Use that URL as `HUNYUAN_REMOTE_URL` for the backend.

## 3. Backend on a VM

On a CPU VM or the same VM:

```bash
export HUNYUAN_REMOTE_URL="https://YOUR_WORKER_8010_URL"
export REPO_URL="https://github.com/TangDien02/AI_3D_Reconstruction_Systerm.git"
bash scripts/gcp_backend_vm_bootstrap.sh
curl http://127.0.0.1:8000/health
```

The script creates a systemd service named `ai-3d-backend`.

## 4. Backend on Cloud Run

Create an Artifact Registry Docker repository once:

```bash
gcloud artifacts repositories create ai-3d-reconstruction \
  --repository-format=docker \
  --location=us-central1
```

Build and push:

```bash
export PROJECT_ID="$(gcloud config get-value project)"
gcloud builds submit \
  --config cloudbuild.yaml \
  --substitutions _IMAGE=us-central1-docker.pkg.dev/$PROJECT_ID/ai-3d-reconstruction/backend:latest
```

Deploy:

```bash
gcloud run deploy ai-3d-backend \
  --image us-central1-docker.pkg.dev/$PROJECT_ID/ai-3d-reconstruction/backend:latest \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 8Gi \
  --cpu 4 \
  --timeout 3600 \
  --set-env-vars RECONSTRUCTION_BACKEND=hunyuan_remote,HUNYUAN_REMOTE_URL=https://YOUR_WORKER_8010_URL,HUNYUAN_REMOTE_OUTPUT_FORMAT=glb,HUNYUAN_REMOTE_ENABLE_TEXTURE=false,HUNYUAN_REMOTE_TIMEOUT_SECONDS=1800,HUNYUAN_REMOTE_POLL_INTERVAL_SECONDS=5
```

Cloud Run storage is ephemeral. For production persistence, copy model outputs
to Cloud Storage or run the backend on a VM with a persistent disk.

## 5. Mobile app

Point Expo to the backend URL:

```bash
cd mobile
npm install
EXPO_PUBLIC_API_BASE_URL="https://YOUR_BACKEND_URL" npm start
```

For a local backend:

```bash
EXPO_PUBLIC_API_BASE_URL="http://YOUR_LAN_IP:8000" npm start
```

## 6. Health checks

Backend:

```bash
curl https://YOUR_BACKEND_URL/health
```

Worker:

```bash
curl https://YOUR_WORKER_8010_URL/health
curl https://YOUR_WORKER_8010_URL/diagnostics
```

## 7. Runtime notes

- Default cloud mode uses `RECONSTRUCTION_BACKEND=hunyuan_remote`.
- Default texture mode is async: shape first, then `/paint-texture`, then poll
  `/texture-jobs/{job_id}`.
- `HUNYUAN_REMOTE_ENABLE_TEXTURE=false` is recommended for T4 and Spot VM runs.
- Use `sudo journalctl -u hunyuan-worker -f` for worker logs.
- Use `sudo journalctl -u ai-3d-backend -f` for VM backend logs.
