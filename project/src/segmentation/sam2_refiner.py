from __future__ import annotations

import importlib.util
import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np
from PIL import Image


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class SAM2Config:
    enabled: bool
    required: bool
    device: str
    model_id: str
    checkpoint: Path | None
    model_cfg: str | None
    multimask_output: bool

    @classmethod
    def from_env(cls) -> "SAM2Config":
        checkpoint_value = os.environ.get("SAM2_CHECKPOINT")
        checkpoint = Path(checkpoint_value).expanduser() if checkpoint_value else None
        enabled = _env_bool("SAM2_ENABLED", False)
        if not enabled and (checkpoint_value or os.environ.get("SAM2_MODEL_ID")):
            enabled = True
        return cls(
            enabled=enabled,
            required=_env_bool("SAM2_REQUIRED", False),
            device=os.environ.get("SAM2_DEVICE", "cuda"),
            model_id=os.environ.get("SAM2_MODEL_ID", "facebook/sam2-hiera-large"),
            checkpoint=checkpoint,
            model_cfg=os.environ.get("SAM2_MODEL_CFG"),
            multimask_output=_env_bool("SAM2_MULTIMASK_OUTPUT", True),
        )

    def readiness(self) -> dict[str, object]:
        sam2_importable = importlib.util.find_spec("sam2") is not None
        checkpoint_exists = None
        if self.checkpoint is not None:
            checkpoint_exists = self.checkpoint.is_file()
        return {
            "enabled": self.enabled,
            "required": self.required,
            "available": bool(self.enabled and sam2_importable and (checkpoint_exists is not False)),
            "sam2_importable": sam2_importable,
            "device": self.device,
            "model_id": self.model_id,
            "checkpoint": str(self.checkpoint) if self.checkpoint else None,
            "checkpoint_exists": checkpoint_exists,
            "model_cfg": self.model_cfg,
            "multimask_output": self.multimask_output,
        }


_predictor = None
_predictor_lock = Lock()


def _select_device(config: SAM2Config) -> str:
    try:
        import torch
    except Exception:
        return "cpu"

    if config.device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return config.device


def _load_predictor(config: SAM2Config):
    global _predictor

    if _predictor is not None:
        return _predictor

    with _predictor_lock:
        if _predictor is not None:
            return _predictor

        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as exc:
            raise RuntimeError(f"SAM2 is not installed/importable: {exc}") from exc

        device = _select_device(config)
        if config.checkpoint is not None or config.model_cfg:
            if config.checkpoint is None or config.model_cfg is None:
                raise RuntimeError("SAM2_CHECKPOINT and SAM2_MODEL_CFG must be set together for local SAM2 loading.")
            if not config.checkpoint.is_file():
                raise RuntimeError(f"SAM2 checkpoint not found: {config.checkpoint}")
            try:
                from sam2.build_sam import build_sam2
            except Exception as exc:
                raise RuntimeError(f"SAM2 local builder is unavailable: {exc}") from exc
            model = build_sam2(config.model_cfg, str(config.checkpoint), device=device)
            _predictor = SAM2ImagePredictor(model)
        else:
            try:
                _predictor = SAM2ImagePredictor.from_pretrained(config.model_id, device=device)
            except TypeError:
                _predictor = SAM2ImagePredictor.from_pretrained(config.model_id)
                if hasattr(getattr(_predictor, "model", None), "to"):
                    _predictor.model.to(device)
        return _predictor


def _inference_context(device: str):
    try:
        import torch
    except Exception:
        return nullcontext()

    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def refine_mask_from_bbox(
    image: Image.Image,
    bbox_xyxy: tuple[float, float, float, float],
    config: SAM2Config | None = None,
) -> tuple[Image.Image, dict[str, object]]:
    config = config or SAM2Config.from_env()
    if not config.enabled:
        raise RuntimeError("SAM2 is disabled.")

    predictor = _load_predictor(config)
    device = _select_device(config)
    image_np = np.asarray(image.convert("RGB"))
    box = np.asarray(bbox_xyxy, dtype=np.float32)

    try:
        import torch
        inference_mode = torch.inference_mode
    except Exception:
        inference_mode = nullcontext

    with inference_mode(), _inference_context(device):
        predictor.set_image(image_np)
        masks, scores, _ = predictor.predict(
            box=box,
            multimask_output=config.multimask_output,
        )

    masks_np = np.asarray(masks)
    if masks_np.ndim == 4:
        masks_np = masks_np.reshape((-1,) + masks_np.shape[-2:])
    elif masks_np.ndim == 2:
        masks_np = masks_np[None, :, :]

    if masks_np.ndim != 3 or masks_np.shape[0] == 0:
        raise RuntimeError(f"SAM2 returned invalid mask shape: {masks_np.shape}")

    scores_np = np.asarray(scores, dtype=np.float32).reshape(-1)
    best_index = int(scores_np.argmax()) if len(scores_np) == len(masks_np) and len(scores_np) > 0 else 0
    selected_mask = masks_np[best_index] > 0
    mask_image = Image.fromarray((selected_mask.astype(np.uint8) * 255), mode="L")
    metadata = {
        "source": "sam2",
        "device": device,
        "model_id": config.model_id,
        "checkpoint": str(config.checkpoint) if config.checkpoint else None,
        "model_cfg": config.model_cfg,
        "box_prompt": [round(float(value), 2) for value in bbox_xyxy],
        "mask_index": best_index,
        "score": float(scores_np[best_index]) if best_index < len(scores_np) else None,
        "num_masks": int(len(masks_np)),
    }
    return mask_image, metadata
