from __future__ import annotations

import torch
from torch import nn


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
