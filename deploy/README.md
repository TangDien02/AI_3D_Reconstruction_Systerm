# Final Deploy Package

Folder nay gom cac file can doc khi VM bi kill hoac can dung lai demo tu dau.

## File chinh

- `FINAL_VM_DEPLOY.ipynb`: notebook chay tren Jupyter/VM de setup worker, check GPU, xem log, tao Cloudflare tunnel.
- `VM_END_TO_END_SETUP.ipynb`: notebook setup VM tu dau de chay ca worker + backend + backend tunnel.
- `APP_FEATURES_AND_USAGE.md`: chuc nang app va cach dung tu mobile.
- `requirements.txt`: dependency cho FastAPI backend + local cleanup + project helper code.
- `COMMANDS.md`: lenh van hanh nhanh cho SSH, Jupyter, Windows backend, Expo, logs, smoke test.
- `scripts/`: script tien ich de chay worker/tunnel/backend theo dung flow hien tai.

## Runtime hien tai

```text
Expo mobile app
-> FastAPI backend /reconstruct-bbox
-> local bbox cleanup (rembg neu co, crop_only fallback)
-> Hunyuan3D worker tren GPU VM
-> GLB tra ve app
```

Khong dung Gemini/Nano Banana API. Khong dung TripoSR.

## Quick path neu VM bi kill

1. Tao lai GPU VM Debian/Ubuntu co NVIDIA driver va CUDA OK.
2. SSH vao VM va clone repo.
3. Chay worker bootstrap:

```bash
cd ~/work/AI_3D_Reconstruction_Systerm
bash scripts/gcp_hunyuan_worker_bootstrap.sh
```

4. Check worker:

```bash
curl http://127.0.0.1:8010/health
sudo journalctl -u hunyuan-worker -f
```

5. Tao Cloudflare tunnel:

```bash
tmux new -s tunnel
cloudflared tunnel --url http://127.0.0.1:8010
```

6. Copy URL `https://....trycloudflare.com` vao `.env.local` cua backend Windows:

```env
HUNYUAN_REMOTE_URL=https://YOUR_NEW_TUNNEL.trycloudflare.com
```

7. Restart backend local va test app.

Neu muon VM chay full end-to-end, mo `VM_END_TO_END_SETUP.ipynb`.
Neu chi muon worker VM + backend Windows, mo `FINAL_VM_DEPLOY.ipynb`.
Chi tiet lenh nhanh nam trong `COMMANDS.md`.
