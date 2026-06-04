from __future__ import annotations

import math
from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class ImageCleanResult:
    clean_image: Image.Image
    input_image: Image.Image
    metadata: dict


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


def _square_pad(image: Image.Image, pad_ratio: float, fill: tuple[int, int, int] = (255, 255, 255)) -> tuple[Image.Image, dict]:
    image = image.convert("RGB")
    width, height = image.size
    pad_ratio = max(0.0, min(float(pad_ratio), 0.5))
    side = max(width, height)
    padded_side = max(1, int(math.ceil(side * (1.0 + 2.0 * pad_ratio))))
    padded = Image.new("RGB", (padded_side, padded_side), fill)
    left = (padded_side - width) // 2
    top = (padded_side - height) // 2
    padded.paste(image, (left, top))
    return padded, {
        "left": left,
        "top": top,
        "right": padded_side - width - left,
        "bottom": padded_side - height - top,
    }


def _prepare_hunyuan_input(image: Image.Image, *, max_side: int, pad_ratio: float) -> tuple[Image.Image, dict]:
    padded, padding = _square_pad(image, pad_ratio)
    resized = _resize_to_max_side(padded, max_side)
    return resized, {
        "pad_ratio": pad_ratio,
        "padding": padding,
        "max_side": max_side,
        "output_size": {
            "width": resized.size[0],
            "height": resized.size[1],
        },
    }


def _rgba_to_white_rgb(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    background.alpha_composite(rgba)
    return background.convert("RGB")


def _trim_to_alpha(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    bbox = alpha.getbbox()
    if not bbox:
        return rgba
    return rgba.crop(bbox)


def _clean_with_rembg(image: Image.Image) -> Image.Image:
    try:
        from rembg import remove
    except Exception as exc:
        raise RuntimeError(f"rembg is not available: {exc}") from exc

    try:
        removed = remove(image.convert("RGBA"))
    except Exception as exc:
        raise RuntimeError(f"rembg failed: {exc}") from exc

    if isinstance(removed, Image.Image):
        rgba = removed.convert("RGBA")
    else:
        raise RuntimeError("rembg returned an unsupported output type.")
    return _rgba_to_white_rgb(_trim_to_alpha(rgba))


def clean_object_image(
    crop_image: Image.Image,
    *,
    backend: str = "auto",
    enable_rembg: bool = True,
    max_side: int = 1536,
    pad_ratio: float = 0.08,
) -> ImageCleanResult:
    requested = (backend or "auto").strip().lower()
    if requested not in {"auto", "rembg", "crop_only"}:
        requested = "auto"

    fallback_chain: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []
    crop_rgb = crop_image.convert("RGB")

    clean_image = None
    cleaner_used = "crop_only"
    if requested in {"auto", "rembg"} and enable_rembg:
        try:
            clean_image = _clean_with_rembg(crop_rgb)
            cleaner_used = "rembg"
            fallback_chain.append("rembg_success")
        except Exception as exc:
            fallback_chain.append("rembg_failed")
            warnings.append("rembg_not_available_or_failed")
            errors.append(str(exc))
    elif requested == "rembg" and not enable_rembg:
        fallback_chain.append("rembg_disabled")
        warnings.append("rembg_disabled")

    if clean_image is None:
        clean_image = crop_rgb
        cleaner_used = "crop_only"
        fallback_chain.append("crop_only_success")

    input_image, input_metadata = _prepare_hunyuan_input(
        clean_image,
        max_side=max_side,
        pad_ratio=pad_ratio,
    )

    return ImageCleanResult(
        clean_image=clean_image,
        input_image=input_image,
        metadata={
            "cleaner_requested": requested,
            "cleaner_used": cleaner_used,
            "fallback_chain": fallback_chain,
            "warnings": warnings,
            "errors": errors,
            "input": input_metadata,
            "sam2_todo": "Optional future backend: user bbox/lasso -> SAM2 mask -> local clean.",
        },
    )
