from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "DejaVu Sans"


def _prepare_output_path(output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _save_current_figure(output_path: str | Path) -> Path:
    output_path = _prepare_output_path(output_path)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def plot_category_distribution(
    data: pd.DataFrame,
    output_path: str | Path,
    category_col: str = "category",
    title: str = "Phân bố số mẫu theo category",
) -> Path:
    """Lưu biểu đồ cột thể hiện số mẫu theo từng category."""
    if category_col not in data.columns:
        raise KeyError(f"Column '{category_col}' not found in dataframe.")

    counts = data[category_col].value_counts().sort_values(ascending=False)

    plt.figure(figsize=(9, 4.8))
    counts.plot(kind="bar", color="#2f6f5e")
    plt.title(title)
    plt.xlabel("Category")
    plt.ylabel("Số mẫu")
    plt.xticks(rotation=45, ha="right")

    return _save_current_figure(output_path)


def plot_cleaning_comparison(
    raw_count: int,
    clean_count: int,
    output_path: str | Path,
    title: str = "Số mẫu trước và sau làm sạch",
) -> Path:
    """Lưu biểu đồ so sánh số mẫu trước và sau làm sạch."""
    labels = ["Ban đầu", "Sau làm sạch"]
    values = [raw_count, clean_count]

    plt.figure(figsize=(5.5, 4.2))
    bars = plt.bar(labels, values, color=["#6f7f89", "#f05a28"])
    plt.title(title)
    plt.ylabel("Số mẫu")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
        )

    return _save_current_figure(output_path)


def plot_missing_files(
    missing_counts: Mapping[str, int],
    output_path: str | Path,
    title: str = "Số lượng file bị thiếu",
) -> Path:
    """Lưu biểu đồ số file ảnh/mask/model bị thiếu."""
    if not missing_counts:
        raise ValueError("missing_counts must not be empty.")

    labels = list(missing_counts.keys())
    values = [int(value) for value in missing_counts.values()]

    plt.figure(figsize=(7, 4.2))
    bars = plt.bar(labels, values, color="#b85c38")
    plt.title(title)
    plt.ylabel("Số mẫu")
    plt.xticks(rotation=20, ha="right")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
        )

    return _save_current_figure(output_path)


def plot_image_size_distribution(
    widths: Sequence[int | float],
    heights: Sequence[int | float],
    output_path: str | Path,
    title: str = "Phân bố kích thước ảnh",
) -> Path:
    """Lưu biểu đồ scatter thể hiện chiều rộng và chiều cao ảnh."""
    if len(widths) != len(heights):
        raise ValueError("widths and heights must have the same length.")
    if len(widths) == 0:
        raise ValueError("widths and heights must not be empty.")

    plt.figure(figsize=(6.4, 4.8))
    plt.scatter(widths, heights, alpha=0.55, color="#1f77b4", edgecolors="none")
    plt.title(title)
    plt.xlabel("Chiều rộng")
    plt.ylabel("Chiều cao")

    return _save_current_figure(output_path)


def plot_baseline_metrics(
    metrics: pd.DataFrame | Sequence[Mapping[str, float | int | str]],
    output_path: str | Path,
    baseline_col: str = "Baseline",
    title: str = "Kết quả baseline",
) -> Path:
    """
    Save grouped bar charts for baseline metrics.

    Expected columns:
    - Baseline
    - Chamfer Distance
    - F-score
    - Precision
    - Recall
    """
    metrics_df = pd.DataFrame(metrics)
    if baseline_col not in metrics_df.columns:
        raise KeyError(f"Column '{baseline_col}' not found in metrics.")

    numeric_cols = [
        col
        for col in metrics_df.columns
        if col != baseline_col and pd.api.types.is_numeric_dtype(metrics_df[col])
    ]
    if not numeric_cols:
        raise ValueError("metrics must contain at least one numeric metric column.")

    ax = metrics_df.set_index(baseline_col)[numeric_cols].plot(
        kind="bar",
        figsize=(9, 4.8),
        width=0.78,
    )
    ax.set_title(title)
    ax.set_xlabel("Baseline")
    ax.set_ylabel("Giá trị metric")
    ax.legend(loc="best")
    plt.xticks(rotation=20, ha="right")

    return _save_current_figure(output_path)


def plot_point_cloud(
    points: np.ndarray | Sequence[Sequence[float]],
    output_path: str | Path,
    title: str = "Point cloud",
    sample_size: int = 2048,
) -> Path:
    """Lưu biểu đồ scatter 3D cho point cloud có dạng [N, 3]."""
    points_np = np.asarray(points, dtype=np.float32)
    if points_np.ndim != 2 or points_np.shape[1] != 3:
        raise ValueError("points must have shape [N, 3].")
    if points_np.shape[0] == 0:
        raise ValueError("points must not be empty.")

    if points_np.shape[0] > sample_size:
        indices = np.linspace(0, points_np.shape[0] - 1, sample_size).astype(int)
        points_np = points_np[indices]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        points_np[:, 0],
        points_np[:, 1],
        points_np[:, 2],
        s=2,
        alpha=0.7,
        color="#2f6f5e",
    )
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    max_range = np.ptp(points_np, axis=0).max()
    center = points_np.mean(axis=0)
    half = max(max_range / 2, 1e-6)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)

    return _save_current_figure(output_path)


def save_dataset_summary_tables(
    raw_data: pd.DataFrame,
    clean_data: pd.DataFrame,
    output_dir: str | Path,
    category_col: str = "category",
) -> dict[str, Path]:
    """Lưu các bảng CSV cần dùng trong báo cáo tuần 2."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(
        [
            {"Thuộc tính": "Tên bộ dữ liệu", "Giá trị": "Pix3D"},
            {"Thuộc tính": "Loại bài toán", "Giá trị": "3D Reconstruction"},
            {"Thuộc tính": "Dữ liệu đầu vào", "Giá trị": "Ảnh RGB + mask"},
            {"Thuộc tính": "Dữ liệu đầu ra", "Giá trị": "Point cloud từ model 3D"},
            {"Thuộc tính": "Số mẫu ban đầu", "Giá trị": len(raw_data)},
            {"Thuộc tính": "Số mẫu sau làm sạch", "Giá trị": len(clean_data)},
            {
                "Thuộc tính": "Số category",
                "Giá trị": clean_data[category_col].nunique()
                if category_col in clean_data.columns
                else "N/A",
            },
            {"Thuộc tính": "Metric", "Giá trị": "Chamfer Distance, F-score"},
        ]
    )
    summary_path = output_dir / "dataset_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    paths = {"dataset_summary": summary_path}

    if category_col in clean_data.columns:
        class_distribution = (
            clean_data[category_col]
            .value_counts()
            .rename_axis("Category")
            .reset_index(name="Số mẫu")
        )
        class_path = output_dir / "class_distribution.csv"
        class_distribution.to_csv(class_path, index=False, encoding="utf-8-sig")
        paths["class_distribution"] = class_path

    return paths


def save_week2_visualizations(
    raw_data: pd.DataFrame,
    clean_data: pd.DataFrame,
    output_dir: str | Path,
    category_col: str = "category",
    missing_counts: Mapping[str, int] | None = None,
    image_sizes: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """
    Save the common week 2 charts and tables.

    image_sizes, when provided, must contain columns: width, height.
    """
    output_dir = Path(output_dir)
    metrics_dir = output_dir / "metrics"
    outputs_dir = output_dir / "outputs"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    paths = save_dataset_summary_tables(
        raw_data=raw_data,
        clean_data=clean_data,
        output_dir=metrics_dir,
        category_col=category_col,
    )

    if category_col in clean_data.columns:
        paths["category_distribution"] = plot_category_distribution(
            clean_data,
            outputs_dir / "category_distribution.png",
            category_col=category_col,
        )

    paths["cleaning_comparison"] = plot_cleaning_comparison(
        raw_count=len(raw_data),
        clean_count=len(clean_data),
        output_path=outputs_dir / "cleaning_comparison.png",
    )

    if missing_counts is not None:
        paths["missing_files"] = plot_missing_files(
            missing_counts,
            outputs_dir / "missing_files.png",
        )

    if image_sizes is not None:
        if "width" not in image_sizes.columns or "height" not in image_sizes.columns:
            raise KeyError("image_sizes must contain 'width' and 'height' columns.")

        paths["image_size_distribution"] = plot_image_size_distribution(
            image_sizes["width"],
            image_sizes["height"],
            outputs_dir / "image_size_distribution.png",
        )

    return paths
