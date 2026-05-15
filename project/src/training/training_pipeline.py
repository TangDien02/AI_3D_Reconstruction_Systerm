from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

# Ghi chu:
# File nay la training pipeline cho bai toan single-view 3D reconstruction.
# Dau vao la anh 2D da duoc dataloader xu ly, dau ra la point cloud 3D du doan.
#
# Luong xu ly:
# 1. Tao Pix3DDataset tu project/data/raw/pix3d.json.
# 2. Chia du lieu thanh train/validation.
# 3. Dung DataLoader tao batch anh va point cloud ground truth.
# 4. Dua anh vao TransformerPointCloudNet de du doan point cloud.
# 5. Tinh Chamfer Distance giua point cloud du doan va ground truth.
# 6. Backpropagation va cap nhat trong so model bang Adam.
# 7. Danh gia bang Chamfer Distance va F-score, sau do luu metric/checkpoint.
#
# Hien tai day la skeleton de chuan bi cho tuan 3, chua phai model toi uu.

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data.dataloader import Pix3DDataset, ProcessedPix3DDataset
from src.metrics.losses import chamfer_distance, f_score


class TransformerPointCloudNet(nn.Module):
    def __init__(
        self,
        num_points: int = 2048,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        mlp_dim: int = 512,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.num_points = num_points
        num_patches = (image_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.decoder = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Linear(1024, num_points * 3),
            nn.Tanh(),
        )

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position_embed, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(images)
        patches = patches.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(images.shape[0], -1, -1)
        tokens = torch.cat([cls_tokens, patches], dim=1)
        tokens = tokens + self.position_embed[:, : tokens.shape[1], :]

        encoded = self.transformer(tokens)
        features = encoded[:, 0]
        points = self.decoder(features)
        return points.view(images.shape[0], self.num_points, 3)


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        images = batch["image"].to(device)
        points_gt = batch["points_gt"].to(device)

        points_pred = model(images)
        loss = chamfer_distance(points_pred, points_gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def evaluate(model, dataloader, device, threshold):
    model.eval()
    total_cd = 0.0
    total_f = 0.0

    for batch in dataloader:
        images = batch["image"].to(device)
        points_gt = batch["points_gt"].to(device)

        points_pred = model(images)
        total_cd += chamfer_distance(points_pred, points_gt).item()
        total_f += f_score(points_pred, points_gt, threshold=threshold)[0]

    num_batches = max(len(dataloader), 1)
    return {
        "chamfer_distance": total_cd / num_batches,
        "f_score": total_f / num_batches,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train single-view 3D point cloud baseline")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir", default="results/training")
    parser.add_argument("--dataset-mode", choices=["raw", "processed"], default="processed")
    parser.add_argument("--split", default="train")
    parser.add_argument("--categories", nargs="+", default=["chair"])
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--num-points", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--transformer-depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--f-threshold", type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()
    raw_dir = (PROJECT_DIR / args.raw_dir).resolve()
    processed_dir = (PROJECT_DIR / args.processed_dir).resolve()
    output_dir = (PROJECT_DIR / args.output_dir).resolve()
    checkpoint_dir = output_dir / "checkpoints"
    metric_dir = output_dir / "metrics"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset_mode == "processed":
        dataset = ProcessedPix3DDataset(
            processed_dir=processed_dir,
            split=args.split,
            categories=args.categories,
            max_samples=args.max_samples,
        )
    else:
        dataset = Pix3DDataset(
            root_dir=raw_dir,
            categories=args.categories,
            image_size=args.image_size,
            num_points=args.num_points,
            max_samples=args.max_samples,
        )

    if len(dataset) < 2:
        raise RuntimeError("Dataset needs at least 2 samples for train/validation split.")

    train_size = max(1, int(len(dataset) * 0.8))
    val_size = len(dataset) - train_size
    if val_size == 0:
        train_size -= 1
        val_size = 1

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = TransformerPointCloudNet(
        num_points=args.num_points,
        image_size=args.image_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.transformer_depth,
        num_heads=args.num_heads,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    metrics_path = metric_dir / "training_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "val_chamfer_distance", "val_f_score"])
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_metrics = evaluate(model, val_loader, device, threshold=args.f_threshold)

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_chamfer_distance": val_metrics["chamfer_distance"],
                "val_f_score": val_metrics["f_score"],
            }
            writer.writerow(row)
            print(
                f"epoch={epoch} "
                f"train_loss={train_loss:.6f} "
                f"val_cd={val_metrics['chamfer_distance']:.6f} "
                f"val_f={val_metrics['f_score']:.4f}"
            )

    checkpoint_path = checkpoint_dir / "transformer_pointcloud_net.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "categories": args.categories,
            "num_points": args.num_points,
            "image_size": args.image_size,
            "patch_size": args.patch_size,
            "embed_dim": args.embed_dim,
            "transformer_depth": args.transformer_depth,
            "num_heads": args.num_heads,
        },
        checkpoint_path,
    )
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
