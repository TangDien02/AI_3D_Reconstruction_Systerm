"""
resnet_encoder.py
-----------------
Load ResNet-18 hoặc ResNet-50 pretrained, bỏ classification head,
trả về feature vector 512-d (ResNet-18) hoặc 2048-d (ResNet-50).

Sử dụng:
    from resnet_encoder import ResNetEncoder

    encoder = ResNetEncoder(backbone="resnet50")
    features = encoder.encode(batch_tensor)   # (N, 2048)
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Literal


class ResNetEncoder(nn.Module):
    """
    ResNet encoder không có classification head.

    Pipeline bên trong:
        conv1 → bn1 → relu → maxpool
        → layer1 → layer2 → layer3 → layer4
        → avgpool → flatten
        → feature vector (N, feature_dim)

    Args:
        backbone   : "resnet18" hoặc "resnet50"
        pretrained : True = load trọng số ImageNet, False = random init
        freeze     : True = đóng băng toàn bộ encoder (chỉ train decoder)
        device     : "cpu" hoặc "cuda"
    """

    BACKBONE_DIM = {
        "resnet18": 512,
        "resnet50": 2048,
    }

    def __init__(
        self,
        backbone:   Literal["resnet18", "resnet50"] = "resnet50",
        pretrained: bool = True,
        freeze:     bool = True,
        device:     str  = "cpu",
    ):
        super().__init__()

        if backbone not in self.BACKBONE_DIM:
            raise ValueError(f"backbone phải là 'resnet18' hoặc 'resnet50', nhận: {backbone}")

        self.backbone    = backbone
        self.feature_dim = self.BACKBONE_DIM[backbone]
        self.device      = torch.device(device)

        # --- Load model pretrained ---
        weights = "IMAGENET1K_V1" if pretrained else None
        if backbone == "resnet18":
            base = models.resnet18(weights=weights)
        else:
            base = models.resnet50(weights=weights)

        # --- Bỏ lớp fc (1000-class classifier), chỉ giữ encoder + avgpool ---
        # base.fc  : Linear(feature_dim → 1000) — ta không cần lớp này
        # base.avgpool : AdaptiveAvgPool2d → (N, C, 1, 1)
        self.encoder = nn.Sequential(
            base.conv1,    # conv 7×7, stride 2  → (N, 64,  H/2,  W/2)
            base.bn1,      # BatchNorm
            base.relu,     # ReLU
            base.maxpool,  # MaxPool 3×3, stride 2 → (N, 64,  H/4,  W/4)
            base.layer1,   # residual block 1       → (N, 256, H/4,  W/4)   [resnet50]
            base.layer2,   # residual block 2       → (N, 512, H/8,  W/8)
            base.layer3,   # residual block 3       → (N, 1024,H/16, W/16)
            base.layer4,   # residual block 4       → (N, 2048,H/32, W/32)
            base.avgpool,  # AdaptiveAvgPool → (N, 2048, 1, 1)
        )

        if freeze:
            self.freeze_all()

        self.to(self.device)

    # ------------------------------------------------------------------ #
    #  Freeze / unfreeze helpers                                           #
    # ------------------------------------------------------------------ #

    def freeze_all(self):
        """Đóng băng toàn bộ encoder — chỉ train decoder ở phase 1."""
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        """Mở toàn bộ encoder để fine-tune."""
        for p in self.encoder.parameters():
            p.requires_grad = True

    def unfreeze_last_n_layers(self, n: int = 2):
        """
        Mở n layer-group cuối của ResNet để fine-tune nhẹ ở phase 2.
        Các layer-group theo thứ tự: conv1/bn1/relu/maxpool, layer1, layer2, layer3, layer4, avgpool.

        Note:
            Only residual blocks are counted; avgpool is ignored because it has no trainable parameters.
            n=1 -> layer4, n=2 -> layer3 + layer4.

        Args:
            n: số layer-group tính từ cuối (thường dùng n=1 hoặc n=2)
        """
        trainable_groups = [
            ("layer1", self.encoder[4]),
            ("layer2", self.encoder[5]),
            ("layer3", self.encoder[6]),
            ("layer4", self.encoder[7]),
        ]

        n = max(0, min(n, len(trainable_groups)))
        if n == 0:
            return

        for _, layer in trainable_groups[-n:]:
            for p in layer.parameters():
                p.requires_grad = True

    def get_trainable_layer_names(self):
        """Return ResNet layer groups that currently have trainable parameters."""
        layer_groups = [
            ("conv1", self.encoder[0]),
            ("bn1", self.encoder[1]),
            ("layer1", self.encoder[4]),
            ("layer2", self.encoder[5]),
            ("layer3", self.encoder[6]),
            ("layer4", self.encoder[7]),
        ]
        return [
            name
            for name, layer in layer_groups
            if any(p.requires_grad for p in layer.parameters())
        ]

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (N, 3, 224, 224) — batch ảnh đã normalize

        Returns:
            features: tensor (N, feature_dim) — feature vector phẳng
        """
        out = self.encoder(x)     # (N, feature_dim, 1, 1)
        return out.flatten(1)     # (N, feature_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Alias của forward() — dùng khi chỉ muốn trích xuất feature,
        không trong training loop.

        Tự động tắt gradient (no_grad) và chuyển input lên đúng device.
        """
        x = x.to(self.device)
        with torch.no_grad():
            return self.forward(x)

    # ------------------------------------------------------------------ #
    #  Info                                                                #
    # ------------------------------------------------------------------ #

    def summary(self):
        """In thông tin model ra console."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = total - trainable
        print(f"Backbone      : {self.backbone}")
        print(f"Feature dim   : {self.feature_dim}")
        print(f"Device        : {self.device}")
        print(f"Total params  : {total:,}")
        print(f"Trainable     : {trainable:,}")
        print(f"Frozen        : {frozen:,}")
        trainable_layers = self.get_trainable_layer_names()
        print(f"Trainable Layers : {trainable_layers if trainable_layers else '(none)'}")


# --------------------------------------------------------------------------- #
#  Quick test                                                                  #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    encoder = ResNetEncoder(backbone="resnet50", pretrained=True, freeze=True)
    encoder.summary()

    dummy = torch.randn(2, 3, 224, 224)         # batch 2 ảnh
    features = encoder.encode(dummy)
    print(f"\nInput  : {dummy.shape}")           # (2, 3, 224, 224)
    print(f"Output : {features.shape}")          # (2, 2048)
