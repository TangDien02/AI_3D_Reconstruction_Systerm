# Pipeline Roadmap

The project is now centered on a TripoSR-first path. Keep the core small and
working before adding more systems.

## Phase 1 - Core

```text
single object image
  -> TripoSR
  -> mesh
  -> sampled point cloud
  -> summary JSON
```

Status: implemented in `src/reconstruction/triposr_runner.py`.

## Phase 2 - Sequential Job Wrapper

```text
validate input
  -> run TripoSR core
  -> write pipeline_manifest.json
```

Status: implemented in `src/pipeline/sequential_3d_pipeline.py`.

## Phase 3 - API

```text
POST /reconstruct-image
  -> direct TripoSR reconstruction

POST /reconstruct-object
  -> YOLO bbox selection
  -> crop object
  -> TripoSR reconstruction
```

Status: implemented in `server/main.py`; requires TripoSR setup and YOLO
weights for object flow.

## Phase 4 - Mobile

```text
camera frame
  -> /detect-frame
  -> choose bbox
  -> /reconstruct-object
  -> open mesh / point cloud artifacts
```

Status: connected in `mobile/App.js`.

## Later

Only add these after the core is stable:

- Better object selection and crop quality checks.
- Batch reconstruction for multiple selected objects.
- Real multi-frame/video pipeline.
- Optional texture baking if the runtime supports OpenGL/xatlas reliably.
- Legacy baseline evaluation only when comparing TripoSR against older models.
