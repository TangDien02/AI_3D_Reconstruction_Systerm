from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.utils.pointcloud_io import save_pointcloud_npy, save_pointcloud_ply


class TripoSRDependencyError(RuntimeError):
    """Raised when the external TripoSR runtime is not importable."""


@dataclass(frozen=True)
class TripoSRConfig:
    model_name_or_path: str = "stabilityai/TripoSR"
    config_name: str = "config.yaml"
    weight_name: str = "model.ckpt"
    triposr_repo_dir: str | None = None
    device: str = "auto"
    chunk_size: int = 8192
    mc_resolution: int = 256
    foreground_ratio: float = 0.85
    remove_background: bool = True
    num_points: int = 2048
    model_save_format: str = "obj"
    normalize_points: bool = True
    seed: int | None = 42


@dataclass(frozen=True)
class TripoSRResult:
    output_dir: Path
    input_path: Path
    processed_input_path: Path
    mesh_path: Path
    colored_mesh_ply_path: Path | None
    pointcloud_npy_path: Path
    pointcloud_ply_path: Path
    preview_path: Path | None
    summary_path: Path
    points: np.ndarray
    summary: dict[str, Any]


def select_device(device_name: str = "auto") -> str:
    requested = (device_name or "auto").lower()
    if requested == "cpu":
        return "cpu"
    if requested not in {"auto", "cuda", "cuda:0"}:
        return device_name
    if torch.cuda.is_available():
        return "cuda:0" if requested in {"auto", "cuda"} else requested
    if requested in {"cuda", "cuda:0"}:
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return "cpu"


def _default_triposr_repo_dir() -> Path:
    return PROJECT_DIR / "external" / "TripoSR"


def _candidate_repo_dirs(config: TripoSRConfig) -> list[Path]:
    candidates: list[Path] = []
    if config.triposr_repo_dir:
        candidates.append(Path(config.triposr_repo_dir))
    env_dir = os.environ.get("TRIPOSR_REPO_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(_default_triposr_repo_dir())
    return candidates


def _import_triposr(config: TripoSRConfig):
    try:
        from tsr.system import TSR

        return TSR
    except Exception as first_exc:
        for repo_dir in _candidate_repo_dirs(config):
            if not repo_dir.is_dir():
                continue
            repo_dir_str = str(repo_dir.resolve())
            if repo_dir_str not in sys.path:
                sys.path.insert(0, repo_dir_str)
            try:
                from tsr.system import TSR

                return TSR
            except Exception:
                continue

        install_hint = (
            "TripoSR runtime is not importable. Clone the official repo to "
            f"{_default_triposr_repo_dir()} or set TRIPOSR_REPO_DIR, then install "
            "project/requirements-triposr.txt. Official repo: "
            "https://github.com/VAST-AI-Research/TripoSR"
        )
        raise TripoSRDependencyError(install_hint) from first_exc


def _write_json(output_path: Path, payload: dict[str, Any]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def _as_uint8_colors(colors: Any, expected_count: int) -> np.ndarray | None:
    if colors is None:
        return None
    colors_np = np.asarray(colors)
    if colors_np.ndim != 2 or colors_np.shape[0] != expected_count or colors_np.shape[1] < 3:
        return None
    colors_np = colors_np[:, :4] if colors_np.shape[1] >= 4 else np.column_stack(
        [colors_np[:, :3], np.full(expected_count, 255)]
    )
    if np.issubdtype(colors_np.dtype, np.floating):
        if colors_np.max(initial=0) <= 1.0:
            colors_np = colors_np * 255.0
        colors_np = np.rint(colors_np)
    return np.clip(colors_np, 0, 255).astype(np.uint8)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def normalize_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    points = points - points.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(points, axis=1).max()
    if scale > 0:
        points = points / scale
    return points.astype(np.float32)


def _repeat_or_trim_points(points: np.ndarray, num_points: int) -> np.ndarray:
    if points.shape[0] == num_points:
        return points.astype(np.float32)
    if points.shape[0] > num_points:
        indices = np.linspace(0, points.shape[0] - 1, num_points).astype(np.int64)
        return points[indices].astype(np.float32)
    repeats = int(np.ceil(num_points / points.shape[0]))
    tiled = np.tile(points, (repeats, 1))
    return tiled[:num_points].astype(np.float32)


def sample_points_from_mesh(
    mesh: Any,
    num_points: int = 2048,
    seed: int | None = 42,
    normalize: bool = True,
) -> np.ndarray:
    if num_points <= 0:
        raise ValueError("num_points must be greater than 0.")
    if not hasattr(mesh, "vertices"):
        raise ValueError("mesh must expose a vertices attribute.")

    vertices = _to_numpy(mesh.vertices).astype(np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
        raise ValueError("mesh vertices must have shape [N, 3] and must not be empty.")

    faces = _to_numpy(getattr(mesh, "faces", np.empty((0, 3), dtype=np.int64)))
    has_faces = faces.ndim == 2 and faces.shape[1] == 3 and faces.shape[0] > 0

    if has_faces:
        import trimesh

        tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces.astype(np.int64), process=False)
        if seed is not None:
            previous_state = np.random.get_state()
            np.random.seed(seed)
        try:
            points, _ = trimesh.sample.sample_surface(tri_mesh, num_points)
        finally:
            if seed is not None:
                np.random.set_state(previous_state)
        points = points.astype(np.float32)
    else:
        points = _repeat_or_trim_points(vertices, num_points=num_points)

    if normalize:
        points = normalize_points(points)
    return points.astype(np.float32)


def save_colored_mesh_ply(mesh: Any, output_path: str | Path) -> Path | None:
    vertices = _to_numpy(getattr(mesh, "vertices", None)).astype(np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
        return None

    vertex_colors = _as_uint8_colors(
        getattr(getattr(mesh, "visual", None), "vertex_colors", None),
        expected_count=vertices.shape[0],
    )
    if vertex_colors is None:
        return None

    faces = _to_numpy(getattr(mesh, "faces", np.empty((0, 3), dtype=np.int64)))
    if faces.ndim != 2 or faces.shape[1] != 3:
        faces = np.empty((0, 3), dtype=np.int64)
    faces = faces.astype(np.int64)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "property uchar alpha",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    lines = header
    for (x, y, z), (r, g, b, a) in zip(vertices, vertex_colors):
        lines.append(f"{x:.7f} {y:.7f} {z:.7f} {int(r)} {int(g)} {int(b)} {int(a)}")
    for a, b, c in faces:
        lines.append(f"3 {int(a)} {int(b)} {int(c)}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _composite_rgba_on_gray(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    image_np = np.asarray(rgba).astype(np.float32) / 255.0
    rgb = image_np[:, :, :3] * image_np[:, :, 3:4] + (1.0 - image_np[:, :, 3:4]) * 0.5
    return Image.fromarray((rgb * 255.0).clip(0, 255).astype(np.uint8))


def prepare_triposr_input(
    image: Image.Image,
    config: TripoSRConfig,
    rembg_session: Any = None,
) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if not config.remove_background:
        if image.mode == "RGBA":
            return _composite_rgba_on_gray(image)
        return image.convert("RGB")

    try:
        _import_triposr(config)
        import rembg
        from tsr.utils import remove_background, resize_foreground
    except Exception as exc:
        raise TripoSRDependencyError(
            "Background removal needs rembg and the TripoSR tsr.utils module. "
            "Install project/requirements-triposr.txt and ensure tsr is importable."
        ) from exc

    session = rembg_session or rembg.new_session()
    rgba = remove_background(image, session)
    rgba = resize_foreground(rgba, config.foreground_ratio)
    return _composite_rgba_on_gray(rgba)


class TripoSRCore:
    def __init__(self, config: TripoSRConfig | None = None, model: Any | None = None):
        self.config = config or TripoSRConfig()
        if self.config.model_save_format not in {"obj", "glb"}:
            raise ValueError("model_save_format must be either 'obj' or 'glb'.")
        self.device = select_device(self.config.device)
        self._model = model

    @property
    def model(self):
        if self._model is None:
            TSR = _import_triposr(self.config)
            model = TSR.from_pretrained(
                self.config.model_name_or_path,
                config_name=self.config.config_name,
                weight_name=self.config.weight_name,
            )
            if hasattr(model, "renderer") and hasattr(model.renderer, "set_chunk_size"):
                model.renderer.set_chunk_size(self.config.chunk_size)
            model.to(self.device)
            self._model = model
        return self._model

    @torch.no_grad()
    def reconstruct_image(
        self,
        image_path: str | Path,
        output_dir: str | Path,
        name: str | None = None,
        save_preview: bool = True,
    ) -> TripoSRResult:
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        if not image_path.is_absolute():
            image_path = PROJECT_DIR / image_path
        if not output_dir.is_absolute():
            output_dir = PROJECT_DIR / output_dir

        output_name = name or image_path.stem
        sample_dir = output_dir / output_name
        sample_dir.mkdir(parents=True, exist_ok=True)

        original = ImageOps.exif_transpose(Image.open(image_path))
        copied_input_path = sample_dir / f"input{image_path.suffix or '.png'}"
        if original.mode in {"RGBA", "LA"} or "transparency" in original.info:
            original.save(copied_input_path)
        else:
            original.convert("RGB").save(copied_input_path)

        processed_input = prepare_triposr_input(original, self.config)
        processed_input_path = sample_dir / "triposr_input.png"
        processed_input.save(processed_input_path)

        scene_codes = self.model([processed_input], device=self.device)
        meshes = self.model.extract_mesh(
            scene_codes,
            True,
            resolution=self.config.mc_resolution,
        )
        if not meshes:
            raise RuntimeError("TripoSR returned no mesh.")
        mesh = meshes[0]
        mesh_vertex_count = int(len(getattr(mesh, "vertices", [])))
        mesh_face_count = int(len(getattr(mesh, "faces", [])))
        vertex_colors = getattr(getattr(mesh, "visual", None), "vertex_colors", None)
        has_vertex_colors = _as_uint8_colors(vertex_colors, mesh_vertex_count) is not None

        mesh_path = sample_dir / f"mesh.{self.config.model_save_format}"
        mesh.export(mesh_path)
        colored_mesh_ply_path = save_colored_mesh_ply(mesh, sample_dir / "mesh_colored.ply")

        points = sample_points_from_mesh(
            mesh,
            num_points=self.config.num_points,
            seed=self.config.seed,
            normalize=self.config.normalize_points,
        )
        npy_path = save_pointcloud_npy(points, sample_dir / "pointcloud.npy")
        ply_path = save_pointcloud_ply(points, sample_dir / "pointcloud.ply")

        preview_path = None
        if save_preview:
            from src.utils.visualization import plot_point_cloud

            preview_path = plot_point_cloud(points, sample_dir / "preview.png", title=output_name)

        summary = {
            "backend": "triposr",
            "input_image": str(image_path),
            "output_name": output_name,
            "output_dir": str(sample_dir),
            "num_points": int(points.shape[0]),
            "points_normalized": bool(self.config.normalize_points),
            "mesh": {
                "format": self.config.model_save_format,
                "vertices": mesh_vertex_count,
                "faces": mesh_face_count,
                "has_vertex_colors": has_vertex_colors,
                "colored_mesh_ply": colored_mesh_ply_path is not None,
            },
            "config": asdict(self.config),
            "runtime": {
                "device": self.device,
                "torch_cuda_available": bool(torch.cuda.is_available()),
            },
            "paths": {
                "copied_input": str(copied_input_path),
                "processed_input": str(processed_input_path),
                "mesh": str(mesh_path),
                "colored_mesh_ply": str(colored_mesh_ply_path) if colored_mesh_ply_path else None,
                "pointcloud_npy": str(npy_path),
                "pointcloud_ply": str(ply_path),
                "preview_png": str(preview_path) if preview_path else None,
            },
        }
        summary_path = _write_json(sample_dir / "triposr_summary.json", summary)

        return TripoSRResult(
            output_dir=sample_dir,
            input_path=copied_input_path,
            processed_input_path=processed_input_path,
            mesh_path=mesh_path,
            colored_mesh_ply_path=colored_mesh_ply_path,
            pointcloud_npy_path=npy_path,
            pointcloud_ply_path=ply_path,
            preview_path=preview_path,
            summary_path=summary_path,
            points=points,
            summary=summary,
        )


def reconstruct_image_to_artifacts(
    image_path: str | Path,
    output_dir: str | Path,
    config: TripoSRConfig | None = None,
    name: str | None = None,
    save_preview: bool = True,
) -> TripoSRResult:
    return TripoSRCore(config=config).reconstruct_image(
        image_path=image_path,
        output_dir=output_dir,
        name=name,
        save_preview=save_preview,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TripoSR core reconstruction for image files.")
    parser.add_argument("--image", nargs="+", required=True, help="Input image path(s).")
    parser.add_argument("--output-dir", default="results/triposr_core", help="Output directory.")
    parser.add_argument("--name", default=None, help="Output name. Only valid for one input image.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or any torch device string.")
    parser.add_argument("--model-name-or-path", default="stabilityai/TripoSR")
    parser.add_argument("--triposr-repo-dir", default=None, help="Path to a local clone of VAST-AI-Research/TripoSR.")
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--mc-resolution", type=int, default=256)
    parser.add_argument("--foreground-ratio", type=float, default=0.85)
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--model-save-format", choices=["obj", "glb"], default="obj")
    parser.add_argument("--no-remove-bg", dest="remove_background", action="store_false")
    parser.set_defaults(remove_background=True)
    parser.add_argument("--no-normalize-points", dest="normalize_points", action="store_false")
    parser.set_defaults(normalize_points=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-preview", dest="save_preview", action="store_false")
    parser.set_defaults(save_preview=True)
    args = parser.parse_args(argv)
    if args.name and len(args.image) != 1:
        parser.error("--name can only be used with a single --image input.")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = TripoSRConfig(
        model_name_or_path=args.model_name_or_path,
        triposr_repo_dir=args.triposr_repo_dir,
        device=args.device,
        chunk_size=args.chunk_size,
        mc_resolution=args.mc_resolution,
        foreground_ratio=args.foreground_ratio,
        remove_background=args.remove_background,
        num_points=args.num_points,
        model_save_format=args.model_save_format,
        normalize_points=args.normalize_points,
        seed=args.seed,
    )
    core = TripoSRCore(config=config)
    results = []
    for image in args.image:
        result = core.reconstruct_image(
            image_path=image,
            output_dir=args.output_dir,
            name=args.name,
            save_preview=args.save_preview,
        )
        results.append(result)
        print(f"Saved TripoSR mesh to {result.mesh_path}")
        print(f"Saved point cloud to {result.pointcloud_ply_path}")
        print(f"Saved summary to {result.summary_path}")

    if len(results) > 1:
        manifest = {
            "backend": "triposr",
            "count": len(results),
            "summaries": [str(result.summary_path) for result in results],
        }
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = PROJECT_DIR / output_dir
        manifest_path = _write_json(output_dir / "triposr_manifest.json", manifest)
        print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
