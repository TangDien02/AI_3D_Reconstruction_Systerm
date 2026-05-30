from __future__ import annotations

"""
evaluate_render_consistency.py
------------------------------
CLI đánh giá Render Consistency Score cho TripoSR outputs.

Hai chế độ:

1. Đánh giá một cặp mesh + ảnh:
   python -m src.evaluation.evaluate_render_consistency \\
       --mesh results/triposr_core/chair/mesh.obj \\
       --image data/processed_2048/images/chair/0001.png \\
       --output-dir results/render_consistency/chair_0001

2. Đánh giá cả thư mục TripoSR output (mỗi subfolder có triposr_summary.json):
   python -m src.evaluation.evaluate_render_consistency \\
       --triposr-dir results/triposr_core \\
       --output-dir results/render_consistency

3. Đánh giá theo CSV manifest (dùng lại fixed_test_samples_chair.csv):
   python -m src.evaluation.evaluate_render_consistency \\
       --manifest benchmarks/fixed_test_samples_chair.csv \\
       --triposr-dir results/triposr_core \\
       --processed-dir data/processed_2048 \\
       --output-dir results/render_consistency_fixed_benchmark

Output mỗi sample:
    <output_dir>/<stem>/
        render_consistency_figure.png   — so sánh input | mask | rendered views
        render_consistency.json         — metrics dict

Output tổng:
    <output_dir>/
        render_consistency_batch.csv    — một dòng mỗi sample
        render_consistency_summary.json — mean/std qua tất cả samples
"""

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.metrics.render_consistency import (
    STANDARD_VIEWPOINTS,
    RenderConsistencyResult,
    compute_render_consistency,
    save_render_consistency_figure,
)


BATCH_CSV_FIELDS = [
    "sample_id",
    "mesh_path",
    "image_path",
    "render_consistency_score",
    "best_ssim",
    "mean_ssim",
    "best_iou",
    "mean_iou",
    "best_view_label",
    "mesh_watertight",
    "mesh_vertices",
    "mesh_faces",
]


# ---------------------------------------------------------------------------
# Single-sample evaluation
# ---------------------------------------------------------------------------

def evaluate_single(
    mesh_path: str | Path,
    image_path: str | Path,
    output_dir: str | Path,
    resolution: int = 224,
    save_figure: bool = True,
    max_figure_views: int = 4,
    sample_id: str | None = None,
) -> dict:
    """
    Đánh giá một cặp (mesh, image).

    Returns dict chứa tất cả metrics để ghi vào CSV / JSON.
    """
    mesh_path = Path(mesh_path)
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = compute_render_consistency(
        mesh_path=mesh_path,
        image_path=image_path,
        resolution=resolution,
    )

    stem = sample_id or mesh_path.parent.name or mesh_path.stem
    metrics_dict = result.to_dict()
    metrics_dict["sample_id"] = stem
    metrics_dict["mesh_path"] = str(mesh_path)
    metrics_dict["image_path"] = str(image_path)

    # Lưu JSON
    json_path = output_dir / "render_consistency.json"
    json_path.write_text(
        json.dumps(metrics_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Lưu hình so sánh
    if save_figure:
        figure_path = output_dir / "render_consistency_figure.png"
        save_render_consistency_figure(
            result=result,
            image_path=image_path,
            output_path=figure_path,
            resolution=resolution,
            max_views=max_figure_views,
        )
        metrics_dict["figure_path"] = str(figure_path)

    print(
        f"[{stem}] score={result.render_consistency_score:.4f} "
        f"best_ssim={result.best_ssim:.4f} best_iou={result.best_iou:.4f} "
        f"best_view={result.best_view_label} "
        f"watertight={result.mesh_watertight} "
        f"faces={result.mesh_faces}"
    )
    return metrics_dict


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def _find_triposr_samples(triposr_dir: Path) -> list[dict]:
    """
    Tìm các TripoSR output samples trong thư mục.
    Mỗi subfolder phải có mesh.obj (hoặc mesh.glb) và
    ít nhất một trong: triposr_input.png, triposr_summary.json.
    """
    samples = []
    for subdir in sorted(triposr_dir.iterdir()):
        if not subdir.is_dir():
            continue

        # Tìm mesh file
        mesh_path = None
        for ext in ("mesh.obj", "mesh.glb"):
            candidate = subdir / ext
            if candidate.is_file():
                mesh_path = candidate
                break
        if mesh_path is None:
            continue

        # Tìm image path (ưu tiên triposr_input.png vì đúng background)
        image_path = None
        for candidate_name in ("triposr_input.png", "input.png", "input.jpg"):
            candidate = subdir / candidate_name
            if candidate.is_file():
                image_path = candidate
                break

        # Nếu không có, đọc từ triposr_summary.json
        if image_path is None:
            summary_path = subdir / "triposr_summary.json"
            if summary_path.is_file():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    raw_input = summary.get("input_image") or summary.get("paths", {}).get("copied_input")
                    if raw_input:
                        candidate = Path(raw_input)
                        if candidate.is_file():
                            image_path = candidate
                except Exception:
                    pass

        if image_path is None:
            print(f"Warning: no input image found for {subdir.name}, skipping.")
            continue

        samples.append({
            "sample_id": subdir.name,
            "mesh_path": mesh_path,
            "image_path": image_path,
        })

    return samples


def _find_samples_from_manifest(
    manifest_path: Path,
    triposr_dir: Path,
    processed_dir: Path,
) -> list[dict]:
    """
    Tìm samples dựa trên fixed benchmark manifest CSV.
    Khớp sample_id giữa manifest và TripoSR output folders.
    """
    import pandas as pd

    manifest = pd.read_csv(manifest_path)
    samples = []

    for _, row in manifest.iterrows():
        sample_id = str(row["sample_id"])
        processed_image = str(row.get("processed_image", ""))

        # Tìm mesh trong triposr_dir (folder tên = sample_id hoặc image stem)
        image_stem = Path(processed_image).stem if processed_image else sample_id
        mesh_path = None
        for candidate_stem in (sample_id, image_stem):
            for ext in ("mesh.obj", "mesh.glb"):
                candidate = triposr_dir / candidate_stem / ext
                if candidate.is_file():
                    mesh_path = candidate
                    break
            if mesh_path is not None:
                break

        if mesh_path is None:
            print(f"Warning: no mesh found for {sample_id} in {triposr_dir}, skipping.")
            continue

        # Tìm image: dùng triposr_input.png nếu có, không thì processed image
        image_path = None
        triposr_input = mesh_path.parent / "triposr_input.png"
        if triposr_input.is_file():
            image_path = triposr_input
        elif processed_image and (processed_dir / processed_image).is_file():
            image_path = processed_dir / processed_image

        if image_path is None:
            print(f"Warning: no image found for {sample_id}, skipping.")
            continue

        samples.append({
            "sample_id": sample_id,
            "mesh_path": mesh_path,
            "image_path": image_path,
        })

    return samples


def _write_batch_csv(rows: list[dict], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=BATCH_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _write_summary_json(rows: list[dict], output_path: Path, extra: dict | None = None) -> Path:
    import numpy as np

    numeric_fields = [
        "render_consistency_score",
        "best_ssim",
        "mean_ssim",
        "best_iou",
        "mean_iou",
    ]
    summary: dict = {"samples": len(rows)}
    for field in numeric_fields:
        values = [float(row[field]) for row in rows if field in row]
        if values:
            summary[f"{field}_mean"] = float(np.mean(values))
            summary[f"{field}_std"] = float(np.std(values))
            summary[f"{field}_min"] = float(np.min(values))
            summary[f"{field}_max"] = float(np.max(values))

    watertight_count = sum(1 for row in rows if row.get("mesh_watertight"))
    summary["mesh_watertight_count"] = watertight_count
    summary["mesh_watertight_ratio"] = watertight_count / max(len(rows), 1)

    if extra:
        summary.update(extra)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def _save_summary_bar_chart(summary_path: Path, output_path: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = {
        k: summary[k]
        for k in [
            "render_consistency_score_mean",
            "best_ssim_mean",
            "best_iou_mean",
            "mean_ssim_mean",
            "mean_iou_mean",
        ]
        if k in summary
    }
    if not metrics:
        return None

    labels = [k.replace("_mean", "").replace("_", "\n") for k in metrics]
    values = list(metrics.values())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    bars = plt.bar(range(len(labels)), values, color="#2f6f5e", alpha=0.85)
    plt.xticks(range(len(labels)), labels, fontsize=8)
    plt.ylabel("Mean value")
    plt.title("Render Consistency — batch summary")
    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    return output_path


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_batch(
    samples: list[dict],
    output_dir: Path,
    resolution: int = 224,
    save_figures: bool = True,
    max_figure_views: int = 4,
    skip_errors: bool = True,
) -> list[dict]:
    """Đánh giá danh sách samples, trả về danh sách metric dicts."""
    rows = []
    total = len(samples)

    for idx, sample in enumerate(samples, start=1):
        sample_id = sample["sample_id"]
        mesh_path = Path(sample["mesh_path"])
        image_path = Path(sample["image_path"])
        sample_output_dir = output_dir / sample_id

        print(f"[{idx}/{total}] {sample_id}")
        try:
            row = evaluate_single(
                mesh_path=mesh_path,
                image_path=image_path,
                output_dir=sample_output_dir,
                resolution=resolution,
                save_figure=save_figures,
                max_figure_views=max_figure_views,
                sample_id=sample_id,
            )
            rows.append(row)
        except Exception as exc:
            msg = f"  ERROR: {exc}"
            print(msg)
            if not skip_errors:
                raise
            rows.append({
                "sample_id": sample_id,
                "mesh_path": str(mesh_path),
                "image_path": str(image_path),
                "render_consistency_score": float("nan"),
                "error": str(exc),
            })

    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Đánh giá Render Consistency Score cho TripoSR mesh outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:

  # Một cặp mesh + ảnh
  python -m src.evaluation.evaluate_render_consistency \\
      --mesh results/triposr_core/chair/mesh.obj \\
      --image data/processed_2048/images/chair/0001.png

  # Cả thư mục TripoSR outputs
  python -m src.evaluation.evaluate_render_consistency \\
      --triposr-dir results/triposr_core \\
      --output-dir results/render_consistency

  # Theo fixed benchmark manifest
  python -m src.evaluation.evaluate_render_consistency \\
      --manifest benchmarks/fixed_test_samples_chair.csv \\
      --triposr-dir results/triposr_core \\
      --processed-dir data/processed_2048 \\
      --output-dir results/render_consistency_fixed
        """,
    )
    # Chế độ single
    parser.add_argument("--mesh", default=None, help="Mesh file (.obj, .glb, .ply)")
    parser.add_argument("--image", default=None, help="Input image (hoặc triposr_input.png)")
    parser.add_argument("--sample-id", default=None, help="Tên định danh sample (tùy chọn)")

    # Chế độ batch
    parser.add_argument(
        "--triposr-dir",
        default=None,
        help="Thư mục chứa TripoSR output subfolders",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path tới fixed benchmark manifest CSV",
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed_2048",
        help="Thư mục data processed (dùng với --manifest)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        default="results/render_consistency",
        help="Thư mục lưu kết quả",
    )
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--no-figures", action="store_true", help="Bỏ qua lưu hình so sánh")
    parser.add_argument("--max-figure-views", type=int, default=4)
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        default=True,
        help="Bỏ qua sample lỗi và tiếp tục (mặc định bật)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = (PROJECT_DIR / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    save_figures = not args.no_figures

    # -----------------------------------------------------------
    # Chế độ single: --mesh + --image
    # -----------------------------------------------------------
    if args.mesh and args.image:
        mesh_path = Path(args.mesh)
        image_path = Path(args.image)
        if not mesh_path.is_absolute():
            mesh_path = PROJECT_DIR / mesh_path
        if not image_path.is_absolute():
            image_path = PROJECT_DIR / image_path

        sample_id = args.sample_id or mesh_path.parent.name or mesh_path.stem
        row = evaluate_single(
            mesh_path=mesh_path,
            image_path=image_path,
            output_dir=output_dir / sample_id,
            resolution=args.resolution,
            save_figure=save_figures,
            max_figure_views=args.max_figure_views,
            sample_id=sample_id,
        )
        print(json.dumps({k: v for k, v in row.items() if k != "per_view"}, indent=2, ensure_ascii=False))
        return

    # -----------------------------------------------------------
    # Chế độ batch
    # -----------------------------------------------------------
    samples: list[dict] = []

    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.is_absolute():
            manifest_path = PROJECT_DIR / manifest_path
        triposr_dir = Path(args.triposr_dir) if args.triposr_dir else None
        if triposr_dir and not triposr_dir.is_absolute():
            triposr_dir = PROJECT_DIR / triposr_dir
        processed_dir = (PROJECT_DIR / args.processed_dir).resolve()

        if triposr_dir is None:
            raise ValueError("--triposr-dir bắt buộc khi dùng --manifest")

        samples = _find_samples_from_manifest(manifest_path, triposr_dir, processed_dir)
        print(f"Manifest: {manifest_path} -> {len(samples)} samples matched.")

    elif args.triposr_dir:
        triposr_dir = Path(args.triposr_dir)
        if not triposr_dir.is_absolute():
            triposr_dir = PROJECT_DIR / triposr_dir
        samples = _find_triposr_samples(triposr_dir)
        print(f"TripoSR dir: {triposr_dir} -> {len(samples)} samples found.")

    else:
        print("Lỗi: cần ít nhất một trong: (--mesh + --image), --triposr-dir, hoặc --manifest")
        return

    if not samples:
        print("Không tìm được sample nào để đánh giá.")
        return

    rows = evaluate_batch(
        samples=samples,
        output_dir=output_dir,
        resolution=args.resolution,
        save_figures=save_figures,
        max_figure_views=args.max_figure_views,
        skip_errors=args.skip_errors,
    )

    # Lưu batch CSV
    valid_rows = [r for r in rows if "render_consistency_score" in r and not _is_nan(r["render_consistency_score"])]
    csv_path = _write_batch_csv(rows, output_dir / "render_consistency_batch.csv")
    print(f"\nSaved batch CSV: {csv_path}")

    # Lưu summary JSON
    summary_path = output_dir / "render_consistency_summary.json"
    _write_summary_json(
        valid_rows,
        summary_path,
        extra={"output_dir": str(output_dir), "total_samples": len(rows), "valid_samples": len(valid_rows)},
    )
    print(f"Saved summary JSON: {summary_path}")

    # Lưu bar chart
    chart_path = _save_summary_bar_chart(summary_path, output_dir / "render_consistency_summary.png")
    if chart_path:
        print(f"Saved summary chart: {chart_path}")

    # Print summary
    if valid_rows:
        import numpy as np
        scores = [r["render_consistency_score"] for r in valid_rows]
        print(
            f"\nSummary ({len(valid_rows)} valid samples):\n"
            f"  render_consistency_score: mean={np.mean(scores):.4f} "
            f"std={np.std(scores):.4f} "
            f"min={np.min(scores):.4f} "
            f"max={np.max(scores):.4f}"
        )


def _is_nan(value: object) -> bool:
    try:
        import math
        return math.isnan(float(value))
    except Exception:
        return False


if __name__ == "__main__":
    main()