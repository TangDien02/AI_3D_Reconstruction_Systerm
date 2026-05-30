from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from src.reconstruction.triposr_runner import TripoSRConfig, TripoSRCore, TripoSRResult


PROJECT_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PipelineStage:
    name: str
    status: str
    started_at: float
    finished_at: float
    duration_ms: float
    detail: str | None = None


@dataclass(frozen=True)
class Sequential3DPipelineConfig:
    backend: str = "triposr"
    output_dir: str = "results/sequential_3d"
    save_preview: bool = True
    triposr: TripoSRConfig = TripoSRConfig()


@dataclass(frozen=True)
class Sequential3DPipelineResult:
    job_id: str
    status: str
    manifest_path: Path
    stages: list[PipelineStage]
    reconstruction: TripoSRResult
    summary: dict[str, Any]


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_DIR / path


def _safe_job_id(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return safe.strip("_") or "reconstruction"


class Sequential3DPipeline:
    """
    A single-process, sequential orchestration layer for image-to-3D jobs.

    This intentionally avoids queues/workers so the current project has one
    debuggable spine first: validate input -> reconstruct -> collect artifacts.
    """

    def __init__(
        self,
        config: Sequential3DPipelineConfig | None = None,
        triposr_core: TripoSRCore | None = None,
    ):
        self.config = config or Sequential3DPipelineConfig()
        if self.config.backend != "triposr":
            raise ValueError("Only the 'triposr' backend is supported by the sequential pipeline.")
        self.triposr_core = triposr_core or TripoSRCore(config=self.config.triposr)

    def _run_stage(self, stages: list[PipelineStage], name: str, fn):
        started = time.perf_counter()
        try:
            result = fn()
        except Exception as exc:
            finished = time.perf_counter()
            stages.append(
                PipelineStage(
                    name=name,
                    status="failed",
                    started_at=started,
                    finished_at=finished,
                    duration_ms=round((finished - started) * 1000, 1),
                    detail=str(exc),
                )
            )
            raise
        finished = time.perf_counter()
        stages.append(
            PipelineStage(
                name=name,
                status="done",
                started_at=started,
                finished_at=finished,
                duration_ms=round((finished - started) * 1000, 1),
            )
        )
        return result

    def run_image(
        self,
        image_path: str | Path,
        job_id: str | None = None,
        output_dir: str | Path | None = None,
    ) -> Sequential3DPipelineResult:
        stages: list[PipelineStage] = []
        image_path = _resolve_project_path(image_path)
        job_id = _safe_job_id(job_id or image_path.stem)
        output_root = _resolve_project_path(output_dir or self.config.output_dir)

        def validate_input() -> dict[str, Any]:
            if not image_path.is_file():
                raise FileNotFoundError(f"Input image not found: {image_path}")
            return {
                "image_path": str(image_path),
                "size_bytes": image_path.stat().st_size,
            }

        input_info = self._run_stage(stages, "validate_input", validate_input)

        def reconstruct() -> TripoSRResult:
            return self.triposr_core.reconstruct_image(
                image_path=image_path,
                output_dir=output_root,
                name=job_id,
                save_preview=self.config.save_preview,
            )

        reconstruction = self._run_stage(stages, "reconstruct_triposr", reconstruct)

        summary = {
            "job_id": job_id,
            "status": "done",
            "backend": self.config.backend,
            "input": input_info,
            "stages": [asdict(stage) for stage in stages],
            "artifacts": {
                "output_dir": str(reconstruction.output_dir),
                "processed_input": str(reconstruction.processed_input_path),
                "mesh": str(reconstruction.mesh_path),
                "colored_mesh_ply": (
                    str(reconstruction.colored_mesh_ply_path)
                    if reconstruction.colored_mesh_ply_path
                    else None
                ),
                "textured_mesh_obj": (
                    str(reconstruction.textured_mesh_obj_path)
                    if reconstruction.textured_mesh_obj_path
                    else None
                ),
                "texture_png": str(reconstruction.texture_path) if reconstruction.texture_path else None,
                "pointcloud_npy": str(reconstruction.pointcloud_npy_path),
                "pointcloud_ply": str(reconstruction.pointcloud_ply_path),
                "preview_png": str(reconstruction.preview_path) if reconstruction.preview_path else None,
                "reconstruction_summary": str(reconstruction.summary_path),
            },
            "reconstruction": reconstruction.summary,
        }
        manifest_path = _write_json(reconstruction.output_dir / "pipeline_manifest.json", summary)
        return Sequential3DPipelineResult(
            job_id=job_id,
            status="done",
            manifest_path=manifest_path,
            stages=stages,
            reconstruction=reconstruction,
            summary=summary,
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the unified sequential image-to-3D pipeline.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--output-dir", default="results/sequential_3d")
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-name-or-path", default="stabilityai/TripoSR")
    parser.add_argument("--triposr-repo-dir", default=None)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--mc-resolution", type=int, default=256)
    parser.add_argument("--foreground-ratio", type=float, default=0.85)
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--model-save-format", choices=["obj", "glb"], default="obj")
    parser.add_argument("--no-remove-bg", dest="remove_background", action="store_false")
    parser.set_defaults(remove_background=True)
    parser.add_argument("--no-preview", dest="save_preview", action="store_false")
    parser.set_defaults(save_preview=True)
    parser.add_argument("--bake-texture", action="store_true")
    parser.add_argument("--texture-resolution", type=int, default=1024)
    parser.add_argument("--texture-padding", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    triposr_config = TripoSRConfig(
        model_name_or_path=args.model_name_or_path,
        triposr_repo_dir=args.triposr_repo_dir,
        device=args.device,
        chunk_size=args.chunk_size,
        mc_resolution=args.mc_resolution,
        foreground_ratio=args.foreground_ratio,
        remove_background=args.remove_background,
        num_points=args.num_points,
        model_save_format=args.model_save_format,
        bake_texture=args.bake_texture,
        texture_resolution=args.texture_resolution,
        texture_padding=args.texture_padding,
    )
    pipeline = Sequential3DPipeline(
        config=Sequential3DPipelineConfig(
            output_dir=args.output_dir,
            save_preview=args.save_preview,
            triposr=triposr_config,
        )
    )
    result = pipeline.run_image(args.image, job_id=args.job_id)
    print(f"Pipeline status: {result.status}")
    print(f"Manifest: {result.manifest_path}")
    print(f"Mesh: {result.reconstruction.mesh_path}")
    print(f"Point cloud: {result.reconstruction.pointcloud_ply_path}")


if __name__ == "__main__":
    main()
