from __future__ import annotations

import shutil
from pathlib import Path


def copy_mesh_asset(source_path: str | Path, output_path: str | Path) -> Path:
    source_path = Path(source_path)
    output_path = Path(output_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Mesh file not found: {source_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() != output_path.resolve():
        shutil.copy2(source_path, output_path)
    return output_path


def summarize_mesh(mesh_path: str | Path) -> dict[str, object]:
    mesh_path = Path(mesh_path)
    summary: dict[str, object] = {
        "path": str(mesh_path),
        "format": mesh_path.suffix.lower().lstrip(".") or "unknown",
        "exists": mesh_path.is_file(),
        "vertices": None,
        "faces": None,
        "surface_reconstruction": mesh_path.is_file(),
    }
    if not mesh_path.is_file():
        return summary

    try:
        import trimesh
    except Exception as exc:
        summary["warning"] = f"trimesh unavailable; skipped mesh stats: {exc}"
        return summary

    try:
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        if isinstance(mesh, trimesh.Scene):
            geometries = tuple(mesh.geometry.values())
            if geometries:
                mesh = trimesh.util.concatenate(geometries)
        summary["vertices"] = int(len(getattr(mesh, "vertices", [])))
        summary["faces"] = int(len(getattr(mesh, "faces", [])))
        summary["is_empty"] = bool(getattr(mesh, "is_empty", False))
    except Exception as exc:
        summary["warning"] = f"failed to inspect mesh: {exc}"

    return summary

