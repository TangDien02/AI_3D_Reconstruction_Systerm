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
Coverage loss: chamfer_gt_weight=1.25
Repulsion loss: repulsion_weight=0.01, repulsion_k=8, repulsion_radius=0.03, repulsion_sample_size=512
```

Co the tinh chinh patience theo validation metric neu can:

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python -m src.training.training_pipeline --dataset-mode processed --categories chair --epochs 100 --batch-size 2 --output-dir results/chair_resnet_baseline --early-stopping-patience 8 --early-stopping-min-delta 0.0001 --early-stopping-min-epochs 12 --lr-scheduler plateau --lr-scheduler-patience 5 --lr-scheduler-factor 0.7
```

Neu point cloud bi thieu coverage hoac bi tum diem, co the tang nhe coverage/repulsion de test:

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python -m src.training.training_pipeline --dataset-mode processed --processed-dir data/processed_2048 --categories chair --epochs 30 --batch-size 16 --max-samples 1024 --val-max-samples 256 --encoder-name resnet50 --feature-dim 2048 --num-points 2048 --output-dir results/bench_repulsion_coverage --device cuda --no-resume --amp --lr-scheduler plateau --freeze-encoder --unfreeze-epoch 6 --augment --chamfer-gt-weight 1.5 --repulsion-weight 0.02 --repulsion-k 8 --repulsion-radius 0.03 --repulsion-sample-size 512
```

Thu nghiem decoder coarse-to-fine refinement de giam point cloud bua:

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python -m src.training.training_pipeline --dataset-mode processed --processed-dir data/processed_2048 --categories chair --epochs 40 --batch-size 16 --encoder-name resnet50 --feature-dim 2048 --decoder-type refine_mlp --coarse-points 512 --refine-offset-scale 0.08 --num-points 2048 --output-dir results/compare_decoder_C_refine_mlp_chair --device cuda --no-resume --amp --lr-scheduler plateau --freeze-encoder --unfreeze-epoch 6 --augment --chamfer-gt-weight 1.5 --repulsion-weight 0.005 --repulsion-k 8 --repulsion-radius 0.03 --repulsion-sample-size 512 --eval-max-samples 128 --comparison-index 0
```

Thu nghiem detail-aware coverage + uniformity loss tren visual baseline decoder:

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python -m src.training.training_pipeline --dataset-mode processed --processed-dir data/processed_2048 --categories chair --epochs 40 --batch-size 16 --encoder-name resnet50 --feature-dim 2048 --decoder-type refine_mlp --coarse-points 512 --refine-offset-scale 0.08 --num-points 2048 --output-dir results/visual_loss_detail_uniform_refine_mlp_chair --device cuda --no-resume --amp --lr-scheduler plateau --freeze-encoder --unfreeze-epoch 6 --augment --chamfer-gt-weight 1.5 --repulsion-weight 0.005 --repulsion-k 8 --repulsion-radius 0.03 --repulsion-sample-size 512 --detail-coverage-weight 0.5 --detail-coverage-k 8 --detail-coverage-sample-size 512 --detail-coverage-max-weight 3.0 --detail-coverage-exponent 1.0 --uniformity-weight 0.003 --uniformity-sample-size 512 --eval-max-samples 128 --comparison-index 0
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

Ngoai metrics chinh thuc `chamfer_distance`, `f_score`, `precision`, `recall`, file CSV/JSON con co visual diagnostic metrics:

```text
fine_f_score / fine_recall   : do chi tiet nho voi threshold chat hon
occupancy_iou                : IoU voxel occupancy giua predict va ground truth
empty_space_violation        : diem predict nam vao vung trong/lo hong cua GT, lower is better
density_score / clump_ratio  : do deu cua point cloud va muc do point bi tum
visual_completeness_score    : diem tong hop visual thong nhat, higher is better
visual_completeness_percent  : visual_completeness_score theo thang 0-100
```

`visual_completeness_score` la metric xep hang chinh cho visual benchmark:

```text
0.30 * surface_alignment_score
+ 0.25 * detail_preservation_score
+ 0.20 * structure_occupancy_score
+ 0.15 * empty_space_score
+ 0.10 * density_uniformity_score
```

Co the tinh chinh nguong diagnostic khi can:

```powershell
python -m src.evaluation.evaluate_baseline --split test --categories chair --batch-size 2 --output-dir results/chair_resnet_baseline --fine-threshold 0.025 --loose-threshold 0.1 --voxel-resolution 32 --occupancy-dilation 1
```

## 6.1 Fixed visual benchmark 10 sample chair

Dung cung mot bo 10 sample test co dinh de so sanh visual giua cac checkpoint:

```text
benchmarks/fixed_test_samples_chair.csv
```

Chay benchmark:

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python -m src.evaluation.evaluate_fixed_visual_benchmark --manifest benchmarks/fixed_test_samples_chair.csv --checkpoint results/all_categories_resnet50_2048pts_30ep_aug/outputs/checkpoints/best_model.pt --output-dir results/fixed_visual_benchmark_baseline --device cuda
```

Output:

```text
results/fixed_visual_benchmark_baseline/
  metrics/fixed_visual_benchmark.csv
  metrics/fixed_visual_benchmark_summary.json
  outputs/fixed_visual_comparison/
    00_pix3d_04666_comparison.png
    ...
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
  <sample_id>_metrics.json
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
$env:RECON_BACKEND="triposr"
$env:TRIPOSR_DIR="C:\models\TripoSR"
$env:TRIPOSR_MODEL_SAVE_FORMAT="glb"
$env:SAM2_ENABLED="false"
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

## 11. Cai TripoSR cho backend main

TripoSR la backend reconstruction chinh. Repo nay chi goi adapter qua `TRIPOSR_DIR`,
khong vendor source TripoSR vao `project/src`.

```powershell
git clone https://github.com/VAST-AI-Research/TripoSR C:\models\TripoSR
cd C:\models\TripoSR
python -m pip install --upgrade setuptools
pip install -r requirements.txt
pip install git+https://github.com/tatsy/torchmcubes.git
```

Chay test truc tiep TripoSR:

```powershell
python run.py examples\chair.png --output-dir output --model-save-format glb
```

Chay backend voi TripoSR:

```powershell
cd <repo>\server
$env:RECON_BACKEND="triposr"
$env:TRIPOSR_DIR="C:\models\TripoSR"
$env:TRIPOSR_MODEL_SAVE_FORMAT="glb"
$env:TRIPOSR_BAKE_TEXTURE="false"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Them SAM2 refine mask sau YOLO:

```powershell
git clone https://github.com/facebookresearch/sam2 C:\models\sam2
cd C:\models\sam2
pip install -e .

cd <repo>\server
$env:SAM2_ENABLED="true"
$env:SAM2_MODEL_ID="facebook/sam2-hiera-large"
$env:SAM2_DEVICE="cuda"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Neu muon fail request khi SAM2 loi thay vi fallback ve YOLO:

```powershell
$env:SAM2_REQUIRED="true"
```

Neu can quay ve baseline point cloud cu:

```powershell
$env:RECON_BACKEND="legacy_pointcloud"
$env:RECON_BASELINE_CHECKPOINT="..\project\results\all_categories_resnet50_2048pts_30ep_aug\outputs\checkpoints\best_model.pt"
```
