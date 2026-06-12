from __future__ import annotations
import math
from PIL import Image, ImageDraw

def clamp_bbox_xyxy(
    xyxy: list[float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    x1 = max(0.0, min(float(x1), float(image_width)))
    y1 = max(0.0, min(float(y1), float(image_height)))
    x2 = max(x1, min(float(x2), float(image_width)))
    y2 = max(y1, min(float(y2), float(image_height)))
    return x1, y1, x2, y2


def bbox_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


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
    margin_ratio: float,
    min_margin_px: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    margin_ratio = max(0.0, min(float(margin_ratio), 0.40))
    margin_x = max(float(min_margin_px), width * margin_ratio)
    margin_y = max(float(min_margin_px), height * margin_ratio)
    return clamp_bbox_xyxy(
        [x1 - margin_x, y1 - margin_y, x2 + margin_x, y2 + margin_y],
        image_width=image_width,
        image_height=image_height,
    )


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


def crop_box_payload(crop_box: tuple[int, int, int, int]) -> dict:
    x1, y1, x2, y2 = crop_box
    return {
        "x": x1,
        "y": y1,
        "width": x2 - x1,
        "height": y2 - y1,
    }


def make_mask_from_polygon(
    polygon: list,
    image_width: int,
    image_height: int,
    fallback_bbox: tuple[float, float, float, float],
) -> Image.Image:
    mask = Image.new("L", (image_width, image_height), 0)
    draw = ImageDraw.Draw(mask)

    if polygon is not None and len(polygon) >= 3:
        points = [(float(x), float(y)) for x, y in polygon]
        draw.polygon(points, fill=255)
    else:
        x1, y1, x2, y2 = fallback_bbox
        draw.rectangle((x1, y1, x2, y2), fill=255)

    return mask

def square_pad_image(
    image: Image.Image,
    fill: tuple[int, int, int] | int = (255, 255, 255),
) -> tuple[Image.Image, dict]:
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
