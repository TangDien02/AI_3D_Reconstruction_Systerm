# TripoSR Core

This branch replaces the project core reconstruction path with TripoSR for local
core testing. The backend and frontend are intentionally not connected yet.

## Install

Run from `project/`:

```powershell
py -3.10 -m venv .venv-triposr
.\.venv-triposr\Scripts\Activate.ps1
.\scripts\install_triposr_core.ps1
```

Use Python 3.10 or 3.11 for this runtime. The existing `.venv-gpu` in this
workspace currently uses Python 3.14, which is too new for the official pinned
TripoSR dependency set.

The install script installs `requirements.txt`, installs
`requirements-triposr.txt`, and clones the official TripoSR repo into:

```text
project/external/TripoSR
```

You can also use an existing clone:

```powershell
$env:TRIPOSR_REPO_DIR="C:\path\to\TripoSR"
```

## Run Core Inference

Run from `project/`:

```powershell
python -m src.reconstruction.triposr_runner --image data\processed_2048\images\chair\0001.png --output-dir results\triposr_core --device auto
```

For already segmented/cropped images with a clean background:

```powershell
python -m src.reconstruction.triposr_runner --image path\to\object.png --output-dir results\triposr_core --no-remove-bg
```

Each input creates:

```text
results/triposr_core/<image_stem>/
  input.<ext>
  triposr_input.png
  mesh.obj
  pointcloud.npy
  pointcloud.ply
  preview.png
  triposr_summary.json
```

## Core Contract

The core output is mesh-first:

```text
image -> TripoSR -> mesh -> sampled point cloud
```

`pointcloud.npy` and `pointcloud.ply` are still exported so evaluation and the
future backend bridge can keep the existing point-cloud contract.

## Local Test Without Downloading TripoSR Weights

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

These tests use a fake TripoSR model and validate the project-level core
artifact contract without downloading model weights.
