from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


WHITE_RGB = (255, 255, 255)


def load_image_safely(image_path: str | Path, mode: str) -> Image.Image:
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    if image.mode == "P" and "transparency" in image.info:
        image = image.convert("RGBA")
    return image.convert(mode)


def clamp_bbox_xyxy(
    xyxy: list[float] | tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    x1 = max(0.0, min(float(x1), float(image_width)))
    y1 = max(0.0, min(float(y1), float(image_height)))
    x2 = max(x1, min(float(x2), float(image_width)))
    y2 = max(y1, min(float(y2), float(image_height)))
    return x1, y1, x2, y2


def parse_bbox_xyxy(value: object, image_size: tuple[int, int]) -> tuple[float, float, float, float]:
    width, height = image_size
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return 0.0, 0.0, float(width), float(height)
    return clamp_bbox_xyxy([float(v) for v in value], image_width=width, image_height=height)


def bbox_to_crop_box(
    box: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    crop_x1 = max(0, min(int(math.floor(x1)), image_width - 1))
    crop_y1 = max(0, min(int(math.floor(y1)), image_height - 1))
    crop_x2 = max(crop_x1 + 1, min(int(math.ceil(x2)), image_width))
    crop_y2 = max(crop_y1 + 1, min(int(math.ceil(y2)), image_height))
    return crop_x1, crop_y1, crop_x2, crop_y2


def crop_box_payload(crop_box: tuple[int, int, int, int]) -> dict[str, int]:
    x1, y1, x2, y2 = crop_box
    return {
        "x": x1,
        "y": y1,
        "width": x2 - x1,
        "height": y2 - y1,
    }


def union_bbox_xyxy(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float]:
    if box_b is None:
        return box_a
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2)


def expand_bbox_xyxy(
    box: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    margin_ratio: float = 0.0,
    min_margin_px: int = 0,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    margin_ratio = max(0.0, min(float(margin_ratio), 0.40))
    margin_x = max(float(min_margin_px), width * margin_ratio) if margin_ratio or min_margin_px else 0.0
    margin_y = max(float(min_margin_px), height * margin_ratio) if margin_ratio or min_margin_px else 0.0
    return clamp_bbox_xyxy(
        [x1 - margin_x, y1 - margin_y, x2 + margin_x, y2 + margin_y],
        image_width=image_width,
        image_height=image_height,
    )


def compose_masked_crop(
    image: Image.Image,
    full_mask: Image.Image,
    crop_box: tuple[int, int, int, int],
    background: tuple[int, int, int] = WHITE_RGB,
) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
    crop = image.crop(crop_box)
    mask_crop = full_mask.crop(crop_box)
    masked_crop = Image.new("RGB", crop.size, background)
    masked_crop.paste(crop, mask=mask_crop)
    transparent_crop = crop.convert("RGBA")
    transparent_crop.putalpha(mask_crop)
    return crop, mask_crop, masked_crop, transparent_crop


def square_pad_image(
    image: Image.Image,
    fill: tuple[int, int, int] | int = WHITE_RGB,
) -> tuple[Image.Image, dict[str, int]]:
    width, height = image.size
    side = max(width, height)
    if image.mode == "L":
        padded = Image.new("L", (side, side), int(fill) if isinstance(fill, int) else 0)
    else:
        padded = Image.new(image.mode, (side, side), fill)
    left = (side - width) // 2
    top = (side - height) // 2
    padded.paste(image, (left, top))
    padding = {
        "left": left,
        "top": top,
        "right": side - width - left,
        "bottom": side - height - top,
    }
    return padded, padding


def preprocess_object_image(
    image: Image.Image,
    mask: Image.Image,
    bbox: object,
    image_size: int = 224,
    margin_ratio: float = 0.0,
    min_margin_px: int = 0,
    background: tuple[int, int, int] = WHITE_RGB,
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    image = image.convert("RGB")
    mask = mask.convert("L").resize(image.size)
    image_width, image_height = image.size

    bbox_xyxy = parse_bbox_xyxy(bbox, image.size)
    mask_bbox = mask.getbbox()
    base_bbox = union_bbox_xyxy(
        bbox_xyxy,
        tuple(float(value) for value in mask_bbox) if mask_bbox else None,
    )
    expanded_bbox = expand_bbox_xyxy(
        base_bbox,
        image_width=image_width,
        image_height=image_height,
        margin_ratio=margin_ratio,
        min_margin_px=min_margin_px,
    )
    crop_box = bbox_to_crop_box(expanded_bbox, image_width=image_width, image_height=image_height)
    _, mask_crop, masked_crop, _ = compose_masked_crop(image, mask, crop_box, background=background)

    padded_image, padding = square_pad_image(masked_crop, fill=background)
    padded_mask, _ = square_pad_image(mask_crop, fill=0)
    processed_image = padded_image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    processed_mask = padded_mask.resize((image_size, image_size), Image.Resampling.NEAREST)

    metadata = {
        "mode": "crop_mask_square_pad_resize",
        "image_size": int(image_size),
        "background": "white",
        "margin_ratio": float(margin_ratio),
        "min_margin_px": int(min_margin_px),
        "base_bbox": crop_box_payload(bbox_to_crop_box(base_bbox, image_width, image_height)),
        "model_crop_bbox": crop_box_payload(crop_box),
        "square_padding": padding,
    }
    return processed_image, processed_mask, metadata
