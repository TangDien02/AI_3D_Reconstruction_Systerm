# App Features And Usage

## App lam gi

AI 3D Reconstruction System bien mot anh chup vat the thanh mesh 3D de xem tren mobile va mo bang Blender.

Flow hien tai:

```text
1. User chup anh trong Expo app
2. User keo bbox quanh vat the
3. App gui anh goc + bbox len FastAPI backend
4. Backend crop bbox va local clean image
5. Backend gui `input.png` sang Hunyuan3D worker tren GPU VM
6. Worker tao shape mesh GLB
7. Backend tra GLB ve app
8. User preview/download/mo GLB
```

## Chuc nang chinh

- Chup anh vat the tu camera mobile.
- Keo bbox de chon dung vat the can reconstruct.
- Crop va clean local tren backend:
  - `rembg` neu co package va model chay duoc.
  - `crop_only` fallback neu `rembg` fail.
- Tao mesh bang Hunyuan3D remote worker.
- Poll job status:
  - `uploaded`
  - `cropping`
  - `cleaning`
  - `generating_shape`
  - `completed`
  - `failed`
- Xem preview artifact backend:
  - `input_original.png`
  - `input_crop.png`
  - `clean_image.png`
  - `input.png`
  - `mesh.glb`

## Cach dung app

1. Dam bao backend local dang chay:

```powershell
curl.exe http://127.0.0.1:8000/health
```

2. Dam bao backend tro toi dung Cloudflare tunnel:

```json
"hunyuan_remote_url": "https://YOUR_TUNNEL.trycloudflare.com"
```

3. Start Expo:

```powershell
cd mobile
$env:EXPO_PUBLIC_API_BASE_URL="http://192.168.1.5:8000"
npm start
```

4. Mo app tren dien thoai cung Wi-Fi.
5. Chup anh vat the ro, nen don gian cang tot.
6. Keo bbox sat quanh vat the, nhung chua cat mat canh.
7. Bam reconstruct va doi job xong.
8. Tai/mo GLB bang Blender neu can check mesh.

## Meo chup anh

- Dung nen don, anh sang ro.
- Vat the nam tron trong bbox.
- Khong de bbox qua rong neu background nhieu chi tiet.
- Neu vat the den/phan xa, nen chup tren nen sang hon.
- Neu mesh bi beo/flat, thu chup goc 3/4 thay vi mat thang.

## Cac endpoint hay dung

```text
GET  /health
POST /preprocess/clean-image
POST /reconstruct-bbox
GET  /reconstruction-jobs/{job_id}
POST /paint-texture
```

`/preprocess/gemini-clean` chi con la alias local backward-compatible va khong goi Gemini.
