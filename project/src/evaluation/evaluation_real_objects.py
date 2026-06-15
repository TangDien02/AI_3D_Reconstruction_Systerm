"""
evaluation_real_objects.py
---------------------------
Evaluate TripoSR meshes on real objects (no ground truth).

Combines:
  - Render consistency (multi-view silhouette match)
  - Mesh quality (watertightness, smoothness)
  - Render quality (depth, normal consistency)

Usage:
    python -m src.evaluation.evaluation_real_objects \\
        --triposr-dir results/triposr_core \\
        --output-dir results/eval_real_objects

Output:
    results/eval_real_objects/
        ├── evaluation_real_objects.csv
        ├── evaluation_real_objects_summary.json
        └── evaluation_real_objects_summary.png
"""

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.metrics.render_consistency import compute_render_consistency
from src.metrics.mesh_quality_metrics import compute_mesh_quality
from src.metrics.render_quality_metrics import compute_render_quality


CSV_FIELDS = [
    "sample_id",
    "mesh_path",
    "image_path",
    # Render consistency
    "render_consistency_score",
    "best_ssim",
    "best_iou",
    "mean_ssim",
    "mean_iou",
    # Mesh quality
    "mesh_quality_score",
    "topology_score",
    "geometry_score",
    "watertight",
    "surface_smoothness",
    # Render quality
    "render_quality_score",
    "depth_consistency_score",
    "normal_consistency_score",
    # Combined
    "overall_quality_score",
]


def find_triposr_samples(triposr_dir: Path) -> list[dict]:
    """Find TripoSR outputs (mesh + triposr_input.png)."""
    samples = []

    for subdir in sorted(triposr_dir.iterdir()):
        if not subdir.is_dir():
            continue

        # Find mesh
        mesh_path = None
        for ext in ("mesh.obj", "mesh.glb"):
            candidate = subdir / ext
            if candidate.is_file():
                mesh_path = candidate
                break

        if mesh_path is None:
            continue

        # Find input image
        image_path = subdir / "triposr_input.png"
        if not image_path.is_file():
            continue

        samples.append({
            "sample_id": subdir.name,
            "mesh_path": mesh_path,
            "image_path": image_path,
        })

    return samples


def evaluate_single(
    sample_id: str,
    mesh_path: Path,
    image_path: Path,
) -> dict | None:
    """Evaluate one sample."""
    try:
        # Render consistency
        rc_result = compute_render_consistency(
            mesh_path=mesh_path,
            image_path=image_path,
        )

        # Mesh quality
        mq_result = compute_mesh_quality(mesh_path)

        # Render quality
        rq_result = compute_render_quality(
            mesh_path=mesh_path,
            input_image_path=image_path,
        )

        # Combined score
        overall_score = (
            0.35 * rc_result.render_consistency_score +
            0.35 * mq_result.mesh_quality_score +
            0.30 * rq_result.render_quality_score
        )

        row = {
            "sample_id": sample_id,
            "mesh_path": str(mesh_path),
            "image_path": str(image_path),
            # Render consistency
            "render_consistency_score": rc_result.render_consistency_score,
            "best_ssim": rc_result.best_ssim,
            "best_iou": rc_result.best_iou,
            "mean_ssim": rc_result.mean_ssim,
            "mean_iou": rc_result.mean_iou,
            # Mesh quality
            "mesh_quality_score": mq_result.mesh_quality_score,
            "topology_score": mq_result.topology_score,
            "geometry_score": mq_result.geometry_score,
            "watertight": mq_result.watertight,
            "surface_smoothness": mq_result.surface_smoothness,
            # Render quality
            "render_quality_score": rq_result.render_quality_score,
            "depth_consistency_score": rq_result.depth_consistency_score,
            "normal_consistency_score": rq_result.normal_consistency_score,
            # Overall
            "overall_quality_score": overall_score,
        }

        print(
            "[{}] render_consistency={:.3f} mesh_quality={:.3f} "
            "render_quality={:.3f} overall={:.3f} watertight={}".format(
                sample_id,
                rc_result.render_consistency_score,
                mq_result.mesh_quality_score,
                rq_result.render_quality_score,
                overall_score,
                mq_result.watertight,
            )
        )

        return row

    except Exception as exc:
        print("[{}] ERROR: {}".format(sample_id, exc))
        return None


def evaluate_batch(samples: list[dict]) -> list[dict]:
    """Batch evaluation."""
    rows = []
    total = len(samples)

    for idx, sample in enumerate(samples, start=1):
        sample_id = sample["sample_id"]
        mesh_path = sample["mesh_path"]
        image_path = sample["image_path"]

        print("[{}/{}] {}".format(idx, total, sample_id), end=" ")
        row = evaluate_single(sample_id, mesh_path, image_path)
        if row:
            rows.append(row)

    return rows


def write_csv(rows: list[dict], output_path: Path) -> Path:
    """Write results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def write_summary_json(rows: list[dict], output_path: Path) -> Path:
    """Write summary statistics."""
    import numpy as np

    numeric_fields = [
        "render_consistency_score",
        "mesh_quality_score",
        "render_quality_score",
        "overall_quality_score",
        "best_ssim",
        "best_iou",
        "surface_smoothness",
    ]

    summary = {"samples": len(rows)}
    for field in numeric_fields:
        values = [float(row[field]) for row in rows if field in row]
        if values:
            summary[f"{field}_mean"] = float(np.mean(values))
            summary[f"{field}_std"] = float(np.std(values))
            summary[f"{field}_min"] = float(np.min(values))
            summary[f"{field}_max"] = float(np.max(values))

    watertight_count = sum(1 for row in rows if row.get("watertight"))
    summary["watertight_count"] = watertight_count
    summary["watertight_ratio"] = watertight_count / max(len(rows), 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate TripoSR meshes on real objects (no-reference)"
    )
    parser.add_argument("--triposr-dir", required=True, help="TripoSR output folder")
    parser.add_argument(
        "--output-dir",
        default="results/eval_real_objects",
        help="Output folder",
    )
    parser.add_argument(
        "--sample-ids",
        default=None,
        help="Comma-separated sample IDs to evaluate (optional filter)",
    )

    args = parser.parse_args()

    triposr_dir = Path(args.triposr_dir)
    output_dir = (PROJECT_DIR / args.output_dir).resolve()

    # Find samples
    samples = find_triposr_samples(triposr_dir)

    if args.sample_ids:
        sample_id_list = args.sample_ids.split(",")
        samples = [s for s in samples if s["sample_id"] in sample_id_list]

    print("Found {} TripoSR samples".format(len(samples)))

    if not samples:
        print("No samples found!")
        return

    # Evaluate
    rows = evaluate_batch(samples)

    if not rows:
        print("No successful evaluations")
        return

    # Write outputs
    csv_path = write_csv(rows, output_dir / "evaluation_real_objects.csv")
    print("Saved CSV: {}".format(csv_path))

    summary_path = write_summary_json(rows, output_dir / "evaluation_real_objects_summary.json")
    print("Saved summary: {}".format(summary_path))

    # Print stats
    import numpy as np

    overall_vals = [row["overall_quality_score"] for row in rows]

    print(
        "\nSummary ({} samples):\n"
        "  Overall quality: mean={:.3f} std={:.3f} min={:.3f} max={:.3f}".format(
            len(rows),
            np.mean(overall_vals),
            np.std(overall_vals),
            np.min(overall_vals),
            np.max(overall_vals),
        )
    )


if __name__ == "__main__":
    main()
