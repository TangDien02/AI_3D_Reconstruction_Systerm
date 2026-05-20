from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.autograd import Function


class GradientReverse(Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, lambda_value: float) -> torch.Tensor:
        ctx.lambda_value = lambda_value
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambda_value * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_value: float = 1.0):
        super().__init__()
        self.lambda_value = lambda_value

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return GradientReverse.apply(inputs, self.lambda_value)


class ConvFeatureEncoder(nn.Module):
    """Small fallback encoder used when torchvision pretrained backbones are unavailable."""

    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images).flatten(1)


class TorchvisionResNetEncoder(nn.Module):
    """Optional ResNet encoder. Requires torchvision, but does not make it a hard dependency."""

    def __init__(self, name: str = "resnet18", pretrained: bool = True, feature_dim: int = 512):
        super().__init__()
        try:
            from torchvision import models
        except Exception as exc:
            raise ImportError("torchvision is required for TorchvisionResNetEncoder") from exc

        if not hasattr(models, name):
            raise ValueError(f"Unsupported torchvision model: {name}")

        builder = getattr(models, name)
        try:
            weights = "DEFAULT" if pretrained else None
            backbone = builder(weights=weights)
        except TypeError:
            backbone = builder(pretrained=pretrained)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projection = nn.Identity() if in_features == feature_dim else nn.Linear(in_features, feature_dim)
        self.feature_dim = feature_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        return self.projection(features)


class AdapterBlock(nn.Module):
    def __init__(self, feature_dim: int, bottleneck_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, feature_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.net(features)


class MLPPointCloudDecoder(nn.Module):
    def __init__(self, feature_dim: int, num_points: int = 512, hidden_dim: int = 1024):
        super().__init__()
        self.num_points = num_points
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_points * 3),
            nn.Tanh(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        points = self.net(features)
        return points.view(features.shape[0], self.num_points, 3)


class DomainDiscriminator(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 256, num_domains: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


@dataclass(frozen=True)
class ReconstructionForwardOutput:
    points: torch.Tensor
    latent: torch.Tensor
    domain_logits: torch.Tensor | None = None


class ObjectReconstructionNet(nn.Module):
    """Object-level encoder-decoder with optional PEFT adapter and ADA branch."""

    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int = 256,
        num_points: int = 512,
        use_adapter: bool = False,
        adapter_bottleneck_dim: int = 64,
        use_domain_discriminator: bool = False,
        grl_lambda: float = 1.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.adapter = AdapterBlock(feature_dim, adapter_bottleneck_dim) if use_adapter else nn.Identity()
        self.decoder = MLPPointCloudDecoder(feature_dim=feature_dim, num_points=num_points)
        self.grl = GradientReversalLayer(lambda_value=grl_lambda)
        self.domain_discriminator = (
            DomainDiscriminator(feature_dim=feature_dim) if use_domain_discriminator else None
        )

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = True

    def trainable_parameter_count(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def forward(self, images: torch.Tensor, return_domain: bool = False) -> ReconstructionForwardOutput:
        latent = self.adapter(self.encoder(images))
        points = self.decoder(latent)
        domain_logits = None
        if return_domain and self.domain_discriminator is not None:
            domain_logits = self.domain_discriminator(self.grl(latent))
        return ReconstructionForwardOutput(points=points, latent=latent, domain_logits=domain_logits)


def build_object_reconstruction_model(
    encoder_name: str = "conv",
    pretrained: bool = True,
    feature_dim: int = 256,
    num_points: int = 512,
    freeze_encoder: bool = True,
    use_adapter: bool = False,
    use_domain_discriminator: bool = False,
) -> ObjectReconstructionNet:
    if encoder_name == "conv":
        encoder = ConvFeatureEncoder(feature_dim=feature_dim)
    elif encoder_name.startswith("resnet"):
        encoder = TorchvisionResNetEncoder(name=encoder_name, pretrained=pretrained, feature_dim=feature_dim)
    else:
        raise ValueError(f"Unsupported encoder_name: {encoder_name}")

    model = ObjectReconstructionNet(
        encoder=encoder,
        feature_dim=feature_dim,
        num_points=num_points,
        use_adapter=use_adapter,
        use_domain_discriminator=use_domain_discriminator,
    )
    if freeze_encoder:
        model.freeze_encoder()
    return model
