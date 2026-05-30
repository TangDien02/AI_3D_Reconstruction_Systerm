# Project Audit

Current focus: **TripoSR is the reconstruction core**.

## Main Path

```text
image or object crop
  -> src.reconstruction.triposr_runner.TripoSRCore
  -> mesh export (.glb or .obj)
  -> sampled point cloud (.npy and .ply)
  -> preview + JSON summary
```

`src.pipeline.sequential_3d_pipeline` wraps this into a simple single-process
job runner. `server/main.py` exposes the same core through FastAPI.

## Active Components

| Path | Role |
| --- | --- |
| `src/reconstruction/triposr_runner.py` | Main TripoSR runner and artifact contract. |
| `src/pipeline/sequential_3d_pipeline.py` | Minimal orchestration around the TripoSR core. |
| `src/utils/pointcloud_io.py` | Saves sampled point cloud outputs. |
| `src/utils/visualization.py` | Saves non-interactive point cloud previews. |
| `tests/test_triposr_runner.py` | Core contract tests with fake TripoSR model. |
| `tests/test_sequential_pipeline.py` | Pipeline manifest contract test. |
| `server/main.py` | YOLO crop + TripoSR reconstruction API. |
| `mobile/App.js` | Expo camera client for server workflow. |

## Legacy Components

These files are retained for comparison/reporting and should not be treated as
the main runtime path:

| Path | Status |
| --- | --- |
| `src/preprocessing/` | Pix3D preprocessing for old baseline training. |
| `src/training/` | ResNet/point-cloud baseline training. |
| `src/evaluation/` | Baseline metric scripts. |
| `src/inference/baseline_inference.py` | Old checkpoint-based point-cloud inference. |
| `src/models/` | Old point-cloud models. |
| `run_all.py`, `main_workflow.py`, `train.py` | Old baseline entry points. |
| `src/pipeline/baseline_runner.py` | Legacy template baseline kept only for reference. |

## Removed / Ignored Noise

- Local `.env` files are ignored and removed from git.
- Generated processed datasets are ignored:
  `project/data/processed/`, `project/data/processed_*`, images, masks,
  `.npy`, `.ply`, `.obj`, `.csv`, and result folders.
- Large processed Pix3D images/masks have been removed from git history at the
  working-tree level and should be regenerated locally when needed.
- The old `tester/` blog CRUD tests were removed because they do not test this
  project domain.

## Current Gaps

- Real TripoSR inference still requires the external official TripoSR clone and
  model weights.
- YOLO object flow requires `server/weights/yolo26n-seg.pt`.
- Video scan endpoints remain mock placeholders; the active core is image/object
  reconstruction.
- Optional texture baking is off by default and depends on native OpenGL/xatlas
  support.
