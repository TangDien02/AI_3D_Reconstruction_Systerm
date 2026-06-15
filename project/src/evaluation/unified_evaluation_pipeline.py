"""
unified_evaluation_pipeline.py
------------------------------
Master evaluation pipeline coordinating all metrics.

Supports two modes:
1. Pix3D benchmark (with ground truth):
   - 3D metrics: Chamfer, Hausdorff, F-score (with ICP)
   - Mesh quality: topology, geometry

2. Real objects (no ground truth):
   - Render consistency: multi-view silhouette
   - Mesh quality: intrinsic metrics
   - Render quality: LPIPS, consistency

Usage:
    # Evaluate on Pix3D
    python -m src.evaluation.unified_evaluation_pipeline \\
        --mode pix3d \\
        --pix3d-dir data/pix3d \\
        --triposr-dir results/triposr_core \\
        --output-dir results/eval_full

    # Evaluate on real objects
    python -m src.evaluation.unified_evaluation_pipeline \\
        --mode real \\
        --triposr-dir results/triposr_core \\
        --output-dir results/eval_real

    # Evaluate both
    python -m src.evaluation.unified_evaluation_pipeline \\
        --mode both \\
        --pix3d-dir data/pix3d \\
        --triposr-dir results/triposr_core \\
        --output-dir results/eval_full
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]


def run_pix3d_evaluation(
    pix3d_dir: Path,
    triposr_dir: Path,
    output_dir: Path,
    category: str | None = None,
) -> dict:
    """Run GT-based Pix3D evaluation."""
    print("\n" + "=" * 70)
    print("PIX3D GROUND TRUTH EVALUATION")
    print("=" * 70)

    cmd = [
        sys.executable,
        "-m",
        "src.evaluation.evaluation_pix3d_cad",
        "--pix3d-dir",
        str(pix3d_dir),
        "--triposr-dir",
        str(triposr_dir),
        "--output-dir",
        str(output_dir / "pix3d_gt"),
    ]

    if category:
        cmd.extend(["--category", category])

    try:
        result = subprocess.run(cmd, cwd=PROJECT_DIR, capture_output=False)
        if result.returncode != 0:
            print("ERROR: Pix3D evaluation failed")
            return {}

        summary_path = output_dir / "pix3d_gt" / "evaluation_pix3d_cad_summary.json"
        if summary_path.is_file():
            with open(summary_path) as f:
                return json.load(f)
    except Exception as exc:
        print(f"ERROR running Pix3D evaluation: {exc}")

    return {}


def run_real_objects_evaluation(
    triposr_dir: Path,
    output_dir: Path,
    sample_ids: str | None = None,
) -> dict:
    """Run no-ref real objects evaluation."""
    print("\n" + "=" * 70)
    print("REAL OBJECTS EVALUATION (NO-REFERENCE)")
    print("=" * 70)

    cmd = [
        sys.executable,
        "-m",
        "src.evaluation.evaluation_real_objects",
        "--triposr-dir",
        str(triposr_dir),
        "--output-dir",
        str(output_dir / "real_objects"),
    ]

    if sample_ids:
        cmd.extend(["--sample-ids", sample_ids])

    try:
        result = subprocess.run(cmd, cwd=PROJECT_DIR, capture_output=False)
        if result.returncode != 0:
            print("ERROR: Real objects evaluation failed")
            return {}

        summary_path = output_dir / "real_objects" / "evaluation_real_objects_summary.json"
        if summary_path.is_file():
            with open(summary_path) as f:
                return json.load(f)
    except Exception as exc:
        print(f"ERROR running real objects evaluation: {exc}")

    return {}


def generate_unified_report(
    output_dir: Path,
    pix3d_summary: dict,
    real_summary: dict,
) -> Path:
    """Generate unified evaluation report."""
    report = {
        "evaluation_mode": "unified",
        "pix3d_gt_benchmark": pix3d_summary or None,
        "real_objects_benchmark": real_summary or None,
        "comparison": {},
    }

    # Comparison
    if pix3d_summary and real_summary:
        report["comparison"] = {
            "note": "Pix3D uses GT-based metrics (Chamfer, F-score). Real objects use no-ref metrics (render consistency, mesh quality).",
            "pix3d_samples": pix3d_summary.get("samples", 0),
            "real_samples": real_summary.get("samples", 0),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "unified_evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    return report_path


def print_summary(
    pix3d_summary: dict,
    real_summary: dict,
):
    """Print evaluation summary."""
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    if pix3d_summary:
        print("\nPix3D Ground Truth Benchmark:")
        print("  Samples: {}".format(pix3d_summary.get("samples", 0)))
        if "chamfer_distance_mean" in pix3d_summary:
            print("  Chamfer distance: {:.4f} +/- {:.4f}".format(
                pix3d_summary["chamfer_distance_mean"],
                pix3d_summary["chamfer_distance_std"],
            ))
        if "f_score_@0.01_mean" in pix3d_summary:
            print("  F-score@0.01: {:.3f} +/- {:.3f}".format(
                pix3d_summary["f_score_@0.01_mean"],
                pix3d_summary["f_score_@0.01_std"],
            ))

    if real_summary:
        print("\nReal Objects Benchmark (no-reference):")
        print("  Samples: {}".format(real_summary.get("samples", 0)))
        if "overall_quality_score_mean" in real_summary:
            print("  Overall quality: {:.3f} +/- {:.3f}".format(
                real_summary["overall_quality_score_mean"],
                real_summary["overall_quality_score_std"],
            ))
        if "watertight_ratio" in real_summary:
            print("  Watertight ratio: {:.1%}".format(real_summary["watertight_ratio"]))

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Unified TripoSR evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Pix3D GT evaluation only
  python -m src.evaluation.unified_evaluation_pipeline \\
      --mode pix3d \\
      --pix3d-dir data/pix3d \\
      --triposr-dir results/triposr_core

  # Real objects (no-ref) evaluation only
  python -m src.evaluation.unified_evaluation_pipeline \\
      --mode real \\
      --triposr-dir results/triposr_core

  # Both benchmarks (comprehensive)
  python -m src.evaluation.unified_evaluation_pipeline \\
      --mode both \\
      --pix3d-dir data/pix3d \\
      --triposr-dir results/triposr_core \\
      --output-dir results/eval_full
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["pix3d", "real", "both"],
        default="both",
        help="Evaluation mode",
    )
    parser.add_argument("--pix3d-dir", help="Pix3D CAD folder (for pix3d mode)")
    parser.add_argument("--triposr-dir", required=True, help="TripoSR output folder")
    parser.add_argument(
        "--output-dir",
        default="results/eval_unified",
        help="Output folder",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Filter Pix3D by category (e.g., chair)",
    )
    parser.add_argument(
        "--sample-ids",
        default=None,
        help="Filter real objects by sample IDs (comma-separated)",
    )

    args = parser.parse_args()

    output_dir = (PROJECT_DIR / args.output_dir).resolve()
    triposr_dir = Path(args.triposr_dir)

    pix3d_summary = {}
    real_summary = {}

    # Run evaluations
    if args.mode in ("pix3d", "both"):
        if not args.pix3d_dir:
            print("ERROR: --pix3d-dir required for pix3d mode")
            return
        pix3d_dir = Path(args.pix3d_dir)
        pix3d_summary = run_pix3d_evaluation(
            pix3d_dir, triposr_dir, output_dir, args.category
        )

    if args.mode in ("real", "both"):
        real_summary = run_real_objects_evaluation(
            triposr_dir, output_dir, args.sample_ids
        )

    # Generate unified report
    report_path = generate_unified_report(output_dir, pix3d_summary, real_summary)
    print(f"\nSaved unified report: {report_path}")

    # Print summary
    print_summary(pix3d_summary, real_summary)


if __name__ == "__main__":
    main()
