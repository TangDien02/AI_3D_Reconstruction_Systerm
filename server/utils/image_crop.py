from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


class ImageCropError(ValueError):
    """Raised when a user supplied crop box cannot produce a useful image."""


@dataclass(frozen=True)
class CropResult:
    crop: Image.Image
    cleaner_input: Image.Image
    metadata: dict


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(float(value), upper))


def _resize_to_max_side(image: Image.Image, max_side: int) -> Image.Image:
    max_side = max(64, int(max_side))
    width, height = image.size
    current_max_side = max(width, height)
    if current_max_side <= max_side:
        return image

    scale = max_side / current_max_side
    next_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return image.resize(next_size, Image.Resampling.LANCZOS)


def _square_pad(image: Image.Image, fill: tuple[int, int, int] = (255, 255, 255)) -> tuple[Image.Image, dict]:
    width, height = image.size
    side = max(width, height)
    padded = Image.new("RGB", (side, side), fill)
    left = (side - width) // 2
    top = (side - height) // 2
    padded.paste(image.convert("RGB"), (left, top))
    return padded, {
        "left": left,
        "top": top,
        "right": side - width - left,
        "bottom": side - height - top,
    }


def crop_user_bbox(
    image: Image.Image,
    bbox_x: float,
    bbox_y: float,
    bbox_width: float,
    bbox_height: float,
    *,
    margin_ratio: float = 0.06,
    min_margin_px: int = 8,
    min_side_px: int = 48,
    max_input_side: int = 2048,
    pad_to_square: bool = True,
) -> CropResult:
    image = image.convert("RGB")
    image_width, image_height = image.size

    if image_width <= 0 or image_height <= 0:
        raise ImageCropError("Input image has invalid dimensions.")

    try:
        x = float(bbox_x)
        y = float(bbox_y)
        width = float(bbox_width)
        height = float(bbox_height)
    except (TypeError, ValueError) as exc:
        raise ImageCropError("bbox fields must be numeric.") from exc

    if width <= 0 or height <= 0:
        raise ImageCropError("bbox width and height must be positive.")

    requested_xyxy = (x, y, x + width, y + height)
    clamped_xyxy = (
        _clamp(requested_xyxy[0], 0, image_width),
        _clamp(requested_xyxy[1], 0, image_height),
        _clamp(requested_xyxy[2], 0, image_width),
        _clamp(requested_xyxy[3], 0, image_height),
    )

    x1, y1, x2, y2 = clamped_xyxy
    clamped_width = x2 - x1
    clamped_height = y2 - y1
    if clamped_width < min_side_px or clamped_height < min_side_px:
        raise ImageCropError(
            f"bbox is too small after clamping. Minimum side is {min_side_px}px."
        )

    margin_ratio = max(0.0, min(float(margin_ratio), 0.35))
    margin_x = max(float(min_margin_px), clamped_width * margin_ratio)
    margin_y = max(float(min_margin_px), clamped_height * margin_ratio)
    expanded_xyxy = (
        _clamp(x1 - margin_x, 0, image_width),
        _clamp(y1 - margin_y, 0, image_height),
        _clamp(x2 + margin_x, 0, image_width),
        _clamp(y2 + margin_y, 0, image_height),
    )

    crop_box = (
        max(0, min(int(expanded_xyxy[0]), image_width - 1)),
        max(0, min(int(expanded_xyxy[1]), image_height - 1)),
        max(1, min(int(round(expanded_xyxy[2])), image_width)),
        max(1, min(int(round(expanded_xyxy[3])), image_height)),
    )
    crop_box = (
        crop_box[0],
        crop_box[1],
        max(crop_box[0] + 1, crop_box[2]),
        max(crop_box[1] + 1, crop_box[3]),
    )

    crop = image.crop(crop_box).convert("RGB")
    cleaner_input = _resize_to_max_side(crop, max_input_side)
    padding = None
    if pad_to_square:
        cleaner_input, padding = _square_pad(cleaner_input)

    metadata = {
        "mode": "user_bbox_crop",
        "image_width": image_width,
        "image_height": image_height,
        "requested_bbox": {
            "x": round(x, 2),
            "y": round(y, 2),
            "width": round(width, 2),
            "height": round(height, 2),
        },
        "clamped_bbox": {
            "x": round(x1, 2),
            "y": round(y1, 2),
            "width": round(clamped_width, 2),
            "height": round(clamped_height, 2),
        },
        "crop_box": {
            "x": crop_box[0],
            "y": crop_box[1],
            "width": crop_box[2] - crop_box[0],
            "height": crop_box[3] - crop_box[1],
        },
        "crop_margin_ratio": margin_ratio,
        "crop_min_margin_px": min_margin_px,
        "cleaner_input_size": {
            "width": cleaner_input.size[0],
            "height": cleaner_input.size[1],
        },
        "cleaner_input_max_side": max_input_side,
        "cleaner_input_square_padding": padding,
    }
    return CropResult(crop=crop, cleaner_input=cleaner_input, metadata=metadata)
