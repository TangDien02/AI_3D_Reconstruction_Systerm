# Technical Commands

Chay cac lenh tu thu muc `project`.

## 1. Cai moi truong Python

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Neu dung server FastAPI:

```powershell
pip install -r ..\server\requirements.txt
```

## 2. Chay preprocessing day du

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python main_workflow.py --skip-training --categories chair --max-samples 256
```

## 3. Train baseline

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python -m src.training.training_pipeline --dataset-mode processed --categories chair --max-samples 256 --epochs 5 --batch-size 4 --output-dir results/baseline
```

Artifact duoc luu vao:

```text
results/baseline/
  logs/baseline.log
  metrics/training_metrics.csv
  outputs/baseline_summary.json
  outputs/training_curves.png
  outputs/checkpoints/transformer_pointcloud_net.pt
```

## 4. Evaluate checkpoint tren test split

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python -m src.evaluation.evaluate_baseline --split test --categories chair --batch-size 4 --output-dir results/baseline
```

Ket qua duoc luu vao:

```text
results/baseline/metrics/test_batch_metrics.csv
results/baseline/metrics/test_summary.json
```

## 5. Inference mot anh va export point cloud

Dung anh da processed:

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python -m src.inference.baseline_inference --image data/processed/images/chair/2003.png --output-dir results/baseline/outputs/inference
```

Output:

```text
results/baseline/outputs/inference/
  <image_name>.npy
  <image_name>.ply
  <image_name>.png
  <image_name>_summary.json
```

## 6. Chay workflow tong

Neu du lieu da preprocessing san:

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python main_workflow.py --skip-preprocessing --categories chair --max-samples 256 --epochs 5 --batch-size 4
```

Neu muon chay lai tu dau:

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python main_workflow.py --categories chair --max-samples 256 --epochs 5 --batch-size 4 --overwrite
```

## 7. Chay backend ky thuat

```powershell
cd ..\server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Kiem tra:

```powershell
curl http://localhost:8000/health
```

Endpoint inference anh:

```powershell
curl -X POST "http://localhost:8000/reconstruct-image" -F "image=@..\project\data\processed\images\chair\2003.png"
```
