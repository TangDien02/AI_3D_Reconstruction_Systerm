"""
evaluation_pix3d_cad.py
-----------------------
Evaluate TripoSR meshes against Pix3D CAD ground truth.

Pipeline:
    Load Pix3D CAD mesh (GT) + TripoSR mesh (prediction)
    → Align with ICP
    → Compute: Chamfer, Hausdorff, F-score, mesh quality
    → Batch process all samples
    → Generate CSV + summary

Usage:
    python -m src.evaluation.evaluation_pix3d_cad \\
        --pix3d-dir data/pix3d \\
        --triposr-dir results/triposr_core \\
        --output-dir results/eval_pix3d_cad

Output:
    results/eval_pix3d_cad/
        ├── evaluation_pix3d_cad.csv       — per-sample metrics
        ├── evaluation_pix3d_cad_summary.json
        └── evaluation_pix3d_cad_summary.png
"""

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.metrics.mesh_3d_metrics import compute_3d_metrics
from src.metrics.mesh_quality_metrics import compute_mesh_quality


CSV_FIELDS = [
    "sample_id",
    "category",
    "mesh_path",
    "gt_path",
    "chamfer_distance",
    "hausdorff_distance",
    "f_score_@0.01",
    "f_score_@0.001",
    "mesh_quality_score",
    "topology_score",
    "geometry_score",
    "watertight",
    "alignment_fitness",
    "scale_ratio",
]


def find_pix3d_samples(pix3d_dir: Path) -> list[dict]:
    """Find Pix3D CAD models organized in category subfolders."""
    samples = []

    for category_dir in sorted(pix3d_dir.iterdir()):
        if not category_dir.is_dir():
            continue

        category = category_dir.name

        # Find CAD mesh files (typically .obj or .glb)
        for mesh_file in category_dir.glob("*.obj"):
            sample_id = mesh_file.stem
            samples.append({
                "sample_id": sample_id,
                "category": category,
                "gt_path": mesh_file,
            })

        for mesh_file in category_dir.glob("*.glb"):
            sample_id = mesh_file.stem
            samples.append({
                "sample_id": sample_id,
                "category": category,
                "gt_path": mesh_file,
            })

    return samples


def find_matching_triposr_mesh(
    sample_id: str,
    category: str,
    triposr_dir: Path,
) -> Path | None:
    """Find TripoSR mesh for a sample."""
    # Try direct match: triposr_dir/sample_id/mesh.*
    for ext in ("mesh.glb", "mesh.obj"):
        candidate = triposr_dir / sample_id / ext
        if candidate.is_file():
            return candidate

    # Try category subfolder: triposr_dir/category/sample_id/mesh.*
    for ext in ("mesh.glb", "mesh.obj"):
        candidate = triposr_dir / category / sample_id / ext
        if candidate.is_file():
            return candidate

    return None


def evaluate_single(
    sample_id: str,
    category: str,
    mesh_path: Path,
    gt_path: Path,
) -> dict | None:
    """Evaluate one sample."""
    try:
        # 3D metrics (with ICP alignment)
        metrics_3d = compute_3d_metrics(
            pred_mesh=mesh_path,
            gt_mesh=gt_path,
            num_samples=50000,
            alignment="icp",
        )

        # Mesh quality
        quality = compute_mesh_quality(mesh_path)

        row = {
            "sample_id": sample_id,
            "category": category,
            "mesh_path": str(mesh_path),
            "gt_path": str(gt_path),
            "chamfer_distance": metrics_3d.chamfer_distance,
            "hausdorff_distance": metrics_3d.hausdorff_distance,
            "f_score_@0.01": metrics_3d.f_score_tau01,
            "f_score_@0.001": metrics_3d.f_score_tau001,
            "mesh_quality_score": quality.mesh_quality_score,
            "topology_score": quality.topology_score,
            "geometry_score": quality.geometry_score,
            "watertight": quality.watertight,
            "alignment_fitness": metrics_3d.alignment_fitness,
            "scale_ratio": metrics_3d.scale_ratio,
        }

        print(
            "[{}] chamfer={:.4f} f@0.01={:.3f} "
            "quality={:.3f} watertight={}".format(
                sample_id,
                metrics_3d.chamfer_distance,
                metrics_3d.f_score_tau01,
                quality.mesh_quality_score,
                quality.watertight,
            )
        )

        return row

    except Exception as exc:
        print("[{}] ERROR: {}".format(sample_id, exc))
        return None


def evaluate_batch(
    pix3d_samples: list[dict],
    triposr_dir: Path,
) -> list[dict]:
    """Batch evaluation."""
    rows = []
    total = len(pix3d_samples)

    for idx, sample in enumerate(pix3d_samples, start=1):
        sample_id = sample["sample_id"]
        category = sample["category"]
        gt_path = sample["gt_path"]

        mesh_path = find_matching_triposr_mesh(sample_id, category, triposr_dir)
        if mesh_path is None:
            print("[{}/{}] {} - TripoSR mesh not found".format(idx, total, sample_id))
            continue

        print("[{}/{}] {}".format(idx, total, sample_id), end=" ")
        row = evaluate_single(sample_id, category, mesh_path, gt_path)
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
        "chamfer_distance",
        "hausdorff_distance",
        "f_score_@0.01",
        "f_score_@0.001",
        "mesh_quality_score",
        "topology_score",
        "geometry_score",
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
        description="Evaluate TripoSR meshes against Pix3D CAD ground truth"
    )
    parser.add_argument("--pix3d-dir", required=True, help="Pix3D CAD folder")
    parser.add_argument("--triposr-dir", required=True, help="TripoSR output folder")
    parser.add_argument(
        "--output-dir",
        default="results/eval_pix3d_cad",
        help="Output folder",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Filter by category (e.g., 'chair')",
    )

    args = parser.parse_args()

    pix3d_dir = Path(args.pix3d_dir)
    triposr_dir = Path(args.triposr_dir)
    output_dir = (PROJECT_DIR / args.output_dir).resolve()

    # Find samples
    pix3d_samples = find_pix3d_samples(pix3d_dir)
    if args.category:
        pix3d_samples = [s for s in pix3d_samples if s["category"] == args.category]

    print("Found {} Pix3D samples".format(len(pix3d_samples)))

    if not pix3d_samples:
        print("No samples found!")
        return

    # Evaluate
    rows = evaluate_batch(pix3d_samples, triposr_dir)

    if not rows:
        print("No successful evaluations")
        return

    # Write outputs
    csv_path = write_csv(rows, output_dir / "evaluation_pix3d_cad.csv")
    print("Saved CSV: {}".format(csv_path))

    summary_path = write_summary_json(rows, output_dir / "evaluation_pix3d_cad_summary.json")
    print("Saved summary: {}".format(summary_path))

    # Print stats
    import numpy as np

    chamfer_vals = [row["chamfer_distance"] for row in rows]
    f_vals = [row["f_score_@0.01"] for row in rows]

    print(
        "\nSummary ({} samples):\n"
        "  Chamfer distance: mean={:.4f} std={:.4f} min={:.4f} max={:.4f}\n"
        "  F-score@0.01: mean={:.4f} std={:.4f} min={:.4f} max={:.4f}".format(
            len(rows),
            np.mean(chamfer_vals),
            np.std(chamfer_vals),
            np.min(chamfer_vals),
            np.max(chamfer_vals),
            np.mean(f_vals),
            np.std(f_vals),
            np.min(f_vals),
            np.max(f_vals),
        )
    )


if __name__ == "__main__":
    main()
