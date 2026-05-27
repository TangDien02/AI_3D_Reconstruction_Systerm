from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.models.object_reconstruction import build_object_reconstruction_model
from src.utils.pointcloud_io import save_pointcloud_npy, save_pointcloud_ply
from src.utils.visualization import plot_point_cloud


def load_image_tensor(
    image_path: str | Path,
    image_size: int,
    input_channels: int = 3,
    mask_path: str | Path | None = None,
) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    image_np = np.asarray(image).astype(np.float32) / 255.0
    image_np = np.transpose(image_np, (2, 0, 1))
    if input_channels == 4:
        if mask_path is not None:
            mask = Image.open(mask_path).convert("L").resize((image_size, image_size))
            mask_np = (np.asarray(mask).astype(np.float32) > 0).astype(np.float32)
        else:
            mask_np = (np.any(image_np[:3] < 0.98, axis=0)).astype(np.float32)
        image_np = np.concatenate([image_np, mask_np[None, :, :]], axis=0)
    return torch.from_numpy(image_np).unsqueeze(0)


def select_device(device_name: str | None = None) -> torch.device:
    requested = (device_name or "auto").lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested not in {"auto", "cuda"}:
        raise ValueError(f"Unsupported device: {device_name}")
    if not torch.cuda.is_available():
        if requested == "cuda":
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        return torch.device("cpu")
    try:
        if torch.cuda.device_count() <= 0:
            raise RuntimeError("torch.cuda.device_count() is 0.")
        torch.empty(1, device="cuda")
        return torch.device("cuda")
    except Exception as exc:
        if requested == "cuda":
            raise RuntimeError(f"CUDA was requested, but it is not usable: {exc}") from exc
        print(f"CUDA is not usable, falling back to CPU: {exc}")
        return torch.device("cpu")


def model_points(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    output = model(images)
    return output.points if hasattr(output, "points") else output


def load_baseline_model(checkpoint_path: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_model_type = checkpoint.get("model_type")
    if checkpoint_model_type not in {None, "resnet_pointcloud"}:
        raise RuntimeError(
            f"Unsupported checkpoint model_type={checkpoint_model_type!r}; expected resnet_pointcloud."
        )
    if checkpoint_model_type is None and any(
        key in checkpoint for key in ("patch_size", "embed_dim", "transformer_depth", "num_heads")
    ):
        raise RuntimeError("This looks like an old Transformer checkpoint. Train a new ResNet checkpoint first.")

    model = build_object_reconstruction_model(
        encoder_name=str(checkpoint.get("encoder_name", "resnet18")),
        pretrained=False,
        normalize_input=bool(checkpoint.get("pretrained", False)),
        feature_dim=int(checkpoint.get("feature_dim", 512)),
        num_points=int(checkpoint.get("num_points", 2048)),
        freeze_encoder=bool(checkpoint.get("freeze_encoder", True)),
        input_channels=int(checkpoint.get("input_channels", 4 if checkpoint.get("use_mask_channel") else 3)),
        decoder_type=str(checkpoint.get("decoder_type", "mlp")),
        coarse_points=int(checkpoint.get("coarse_points", 512)),
        refine_offset_scale=float(checkpoint.get("refine_offset_scale", 0.08)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def predict_pointcloud(
    image_path: str | Path,
    checkpoint_path: str | Path,
    device: torch.device | None = None,
    mask_path: str | Path | None = None,
) -> tuple[np.ndarray, dict]:
    device = device or select_device()
    model, checkpoint = load_baseline_model(checkpoint_path, device=device)
    image_size = int(checkpoint.get("image_size", 224))
    input_channels = int(checkpoint.get("input_channels", 4 if checkpoint.get("use_mask_channel") else 3))
    image_tensor = load_image_tensor(
        image_path,
        image_size=image_size,
        input_channels=input_channels,
        mask_path=mask_path,
    ).to(device)
    points = model_points(model, image_tensor).squeeze(0).cpu().numpy().astype(np.float32)
    return points, checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline point cloud inference for one image.")
    parser.add_argument("--image", required=True, help="Input RGB image path.")
    parser.add_argument(
        "--checkpoint",
        default="results/all_categories_resnet50_2048pts_30ep_aug/outputs/checkpoints/best_model.pt",
        help="Baseline checkpoint path relative to project/ or absolute.",
    )
    parser.add_argument("--output-dir", default="results/all_categories_resnet50_2048pts_30ep_aug/outputs/inference")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--name", default=None, help="Optional output basename.")
    parser.add_argument("--mask", default=None, help="Optional binary mask path for 4-channel checkpoints.")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    if not image_path.is_absolute():
        image_path = PROJECT_DIR / image_path
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_DIR / checkpoint_path
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir

    output_name = args.name or image_path.stem
    mask_path = Path(args.mask) if args.mask else None
    if mask_path is not None and not mask_path.is_absolute():
        mask_path = PROJECT_DIR / mask_path
    points, checkpoint = predict_pointcloud(
        image_path,
        checkpoint_path,
        device=select_device(args.device),
        mask_path=mask_path,
    )

    npy_path = save_pointcloud_npy(points, output_dir / f"{output_name}.npy")
    ply_path = save_pointcloud_ply(points, output_dir / f"{output_name}.ply")
    plot_path = None
    if not args.no_plot:
        plot_path = plot_point_cloud(points, output_dir / f"{output_name}.png", title=output_name)

    summary_path = output_dir / f"{output_name}_summary.json"
    summary = {
        "image_path": str(image_path),
        "checkpoint_path": str(checkpoint_path),
        "mask_path": str(mask_path) if mask_path else None,
        "categories": checkpoint.get("categories"),
        "input_channels": checkpoint.get("input_channels", 4 if checkpoint.get("use_mask_channel") else 3),
        "use_mask_channel": checkpoint.get("use_mask_channel", False),
        "decoder_type": checkpoint.get("decoder_type", "mlp"),
        "coarse_points": checkpoint.get("coarse_points"),
        "refine_offset_scale": checkpoint.get("refine_offset_scale"),
        "num_points": int(points.shape[0]),
        "npy_path": str(npy_path),
        "ply_path": str(ply_path),
        "plot_path": str(plot_path) if plot_path else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved point cloud NPY to {npy_path}")
    print(f"Saved point cloud PLY to {ply_path}")
    print(f"Saved inference summary to {summary_path}")
    if plot_path:
        print(f"Saved point cloud preview to {plot_path}")


if __name__ == "__main__":
    main()

