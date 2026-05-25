# Technical Commands

Chay cac lenh tu thu muc `project`.

Tai lieu audit chi tiet ve folder/file/function/thuat toan nam tai:

```text
PROJECT_AUDIT.md
```

Luu y: pipeline hien tai la **single image -> point cloud**, khong phai video 360 -> mesh `.glb`. Cac endpoint video/detection trong `server` hien van la mock.

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

## 2. Mo giao dien cau hinh training

Dung giao dien nay neu muon chon category, output folder, train tiep tu `best_model.pt`, train lai tu dau, hoac chon checkpoint tuy chon. Giao dien se bao loi neu checkpoint khac category hoac khac kien truc model.

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python main_workflow.py --skip-training --categories chair --overwrite
.\.venv-gpu\Scripts\python.exe train_config_gui.py
```

Neu may local bi loi CUDA hoac dung GTX 1050 voi ban PyTorch khong tuong thich, tick `Force CPU`. Khi do lenh train se co them `--device cpu`.

Sau khi train xong, pipeline tu dong tao cac file truc quan va bang sau:

```text
metrics/training_metrics.csv
metrics/test_summary.json
metrics/test_batch_metrics.csv
outputs/training_curves.png
outputs/test_summary_metrics.png
outputs/test_batch_metrics.png
outputs/comparison/*_comparison.png
```

Che do checkpoint:

```text
Auto: dung best_model.pt neu co      tiep tuc train tu best_model.pt trong output dir; neu chua co thi train moi
Train model moi                      khong resume; can can than vi co the ghi de checkpoint cu trong output dir
Bat buoc resume best_model.pt         chi chay neu best_model.pt ton tai va dung cau hinh
Resume resnet_pointcloud_net.pt       resume tu checkpoint epoch cuoi
Checkpoint tuy chon                  chon mot file .pt bat ky
```

## 3. Chay nhanh smoke test end-to-end

Lenh nay chay preprocessing nho, train 1 epoch, evaluate test split va tao anh so sanh point cloud predict voi ground truth.

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python -m src.training.training_pipeline --dataset-mode processed --categories chair --epochs 5 --batch-size 4 --output-dir results/baseline
python run_all.py --quick --no-resume
```

Ket qua smoke test nam trong:

```text
results/smoke_test/
  logs/baseline.log
  metrics/training_metrics.csv
  metrics/test_batch_metrics.csv
  metrics/test_summary.json
  outputs/checkpoints/best_model.pt
  outputs/checkpoints/resnet_pointcloud_net.pt
  outputs/comparison/<sample_id>_comparison.png
  outputs/comparison/<sample_id>_pred.npy
  outputs/comparison/<sample_id>_gt.npy
  outputs/run_all_summary.json
  outputs/training_curves.png
```

## 4. Chay preprocessing day du

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python main_workflow.py --skip-training --categories chair
```

## 5. Train baseline theo category

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python -m src.training.training_pipeline --dataset-mode processed --categories chair --epochs 5 --batch-size 2 --output-dir results/chair_resnet_baseline
```

Mac dinh lenh train se tu dong resume tu checkpoint:

```text
results/<category>_resnet_baseline/outputs/checkpoints/best_model.pt
```

Neu checkpoint nay ton tai, lan train tiep theo se load `best_model.pt` va train tiep them so epoch duoc truyen trong `--epochs`. Neu muon train lai tu dau va bo qua checkpoint cu, them `--no-resume`:

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python -m src.training.training_pipeline --dataset-mode processed --categories chair --epochs 5 --batch-size 2 --output-dir results/chair_resnet_baseline --no-resume
```

Neu muon resume tu mot checkpoint cu the, dung `--resume-checkpoint`:

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python -m src.training.training_pipeline --dataset-mode processed --categories chair --epochs 5 --batch-size 2 --output-dir results/chair_resnet_baseline --resume-checkpoint results/chair_resnet_baseline/outputs/checkpoints/best_model.pt
```

Neu chi can smoke test nhanh, them `--max-samples 256`.

Early stopping duoc bat mac dinh voi `patience=8`, `min_delta=0.0001`, `min_epochs=12`.
AMP va ReduceLROnPlateau cung duoc bat mac dinh theo cau hinh an toan:

```text
AMP: tu dong bat khi device=cuda, tu dong tat khi device=cpu
ReduceLROnPlateau: factor=0.7, patience=5, threshold=0.0001, min_lr=0.000001
```

Co the tinh chinh patience theo validation metric neu can:

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python -m src.training.training_pipeline --dataset-mode processed --categories chair --epochs 100 --batch-size 2 --output-dir results/chair_resnet_baseline --early-stopping-patience 8 --early-stopping-min-delta 0.0001 --early-stopping-min-epochs 12 --lr-scheduler plateau --lr-scheduler-patience 5 --lr-scheduler-factor 0.7
```

Artifact duoc luu vao:

```text
results/<category>_resnet_baseline/
  logs/baseline.log
  metrics/training_metrics.csv
  outputs/baseline_summary.json
  outputs/training_curves.png
  outputs/checkpoints/best_model.pt
  outputs/checkpoints/resnet_pointcloud_net.pt
```

## 6. Evaluate checkpoint tren test split

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python -m src.evaluation.evaluate_baseline --split test --categories chair --batch-size 2 --output-dir results/chair_resnet_baseline
```

Ket qua duoc luu vao:

```text
results/chair_resnet_baseline/metrics/test_batch_metrics.csv
results/chair_resnet_baseline/metrics/test_summary.json
```

## 7. So sanh point cloud predict voi ground truth

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python -m src.inference.compare_pointclouds --split test --categories chair --output-dir results/chair_resnet_baseline/outputs/comparison
```

Output:

```text
results/chair_resnet_baseline/outputs/comparison/
  <sample_id>_comparison.png
  <sample_id>_pred.npy
  <sample_id>_gt.npy
```

## 8. Inference mot anh va export point cloud

Dung anh da processed:

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python -m src.inference.baseline_inference --image data/processed/images/chair/2003.png --output-dir results/chair_resnet_baseline/outputs/inference
```

Output:

```text
results/chair_resnet_baseline/outputs/inference/
  <image_name>.npy
  <image_name>.ply
  <image_name>.png
  <image_name>_summary.json
```

## 6. So sanh point cloud du doan voi ground truth

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python -m src.inference.compare_pointclouds --checkpoint results/baseline/outputs/checkpoints/transformer_pointcloud_net.pt --processed-dir data/processed --split val --categories chair --index 0 --output-dir results/baseline/outputs/comparison
```

Output:

```text
results/baseline/outputs/comparison/
  <sample_id>_pred.npy
  <sample_id>_gt.npy
  <sample_id>_comparison.png
```

## 7. Chay workflow tong
## 9. Chay workflow tong

Neu du lieu da preprocessing san:

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python main_workflow.py --skip-preprocessing --categories chair --epochs 5 --batch-size 4
python main_workflow.py --skip-preprocessing --categories chair --epochs 5 --batch-size 2
```

Neu muon chay lai tu dau:

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python main_workflow.py --categories chair --epochs 5 --batch-size 4 --overwrite
```

## 8. Chay backend ky thuat
python main_workflow.py --categories chair --epochs 5 --batch-size 2 --overwrite
```

Neu muon chay tron goi giong project phu, dung:

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python run_all.py --category chair --epochs 5 --batch-size 2 --output-dir results/chair_resnet_baseline
```

## 10. Chay backend ky thuat

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
