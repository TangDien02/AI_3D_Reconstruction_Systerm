# from src.utils.logger import get_logger

# logger = get_logger("ImagePreprocessor")

# class ImagePreprocessor:
#     def __init__(self):
#         self.yolo_model = None
#         try:
#             from ultralytics import YOLO

#             logger.info("Khởi tạo mô hình YOLOv8-seg (Fallback)...")
#             self.yolo_model = YOLO('yolov8n-seg.pt')
#         except Exception as e:
#             logger.warning(f"Không thể khởi tạo YOLO: {e}")

#     def get_class_from_pre_data(self, image_id, metadata_dict):
#         """ Lấy class từ pre-data (như file metadata JSON) """
#         if metadata_dict and image_id in metadata_dict:
#             return metadata_dict[image_id]
#         return None

#     def process(self, image_path, pre_data_class=None):
#         """ Logic kết hợp: Ưu tiên Pre-Data, Fallback dùng YOLO """
#         if pre_data_class:
#             logger.info(f"Sử dụng class từ Pre-Data: {pre_data_class}")
#             return pre_data_class

#         if self.yolo_model:
#             logger.info(f"Chạy YOLO Infer cho: {image_path}")
#             results = self.yolo_model(image_path, verbose=False)
#             if len(results[0].boxes) > 0:
#                 box = results[0].boxes[0]
#                 cls_id = int(box.cls[0])
#                 return self.yolo_model.names[cls_id]
                
#         return None




"""
image_processor.py
------------------
Đọc ảnh từ file hoặc nhận PIL Image / numpy array,
áp dụng transform cho ảnh object đã được crop/mask sẵn từ YOLO/Backend,
và trả về tensor sẵn sàng đưa vào ResNet encoder.

Pipeline (thay thế Resize→CenterCrop cũ):
    pad_square → Resize((224, 224)) → ToTensor → Normalize(ImageNet)

Lý do đổi pipeline:
    - Ảnh đầu vào là object đã crop/mask từ YOLO — không cần CenterCrop.
    - CenterCrop có thể cắt mất mép vật thể nếu object không nằm chính giữa.
    - pad_square giữ toàn bộ vật thể, tránh méo hình khi resize về 224×224.

Sử dụng:
    from image_processor import ImageProcessor

    processor = ImageProcessor()
    tensor = processor.process("obj_crop.jpg")              # (1, 3, 224, 224)
    batch  = processor.process_batch(["a.jpg", "b.jpg"])    # (2, 3, 224, 224)
"""

from pathlib import Path
from typing import List, Union

import torch
from PIL import Image, ImageOps
import torchvision.transforms as transforms


# Kiểu dữ liệu đầu vào được chấp nhận
ImageInput = Union[str, Path, Image.Image]


def pad_to_square(pil_img: Image.Image, fill: int = 0) -> Image.Image:
    """
    Pad ảnh về hình vuông bằng cách thêm viền đều 2 phía (cạnh ngắn hơn).
    Giữ toàn bộ nội dung gốc, không cắt bất kỳ pixel nào.

    Args:
        pil_img : PIL Image đầu vào (bất kỳ tỉ lệ nào)
        fill    : giá trị pixel fill cho vùng padding (mặc định 0 = đen)

    Returns:
        PIL Image hình vuông, cạnh = max(W, H) của ảnh gốc
    """
    w, h = pil_img.size
    if w == h:
        return pil_img

    side = max(w, h)
    # ImageOps.pad center-aligns và pad đều 2 phía
    return ImageOps.pad(pil_img, (side, side), color=fill)


class ImageProcessor:
    """
    Chuẩn bị ảnh object (đã crop/mask từ YOLO) cho ResNet-50 encoder.

    Pipeline:
        1. pad_square      — đệm ảnh về hình vuông, giữ toàn bộ vật thể
        2. Resize(224,224) — resize về đúng kích thước model
        3. ToTensor        — H×W×3 uint8 [0,255] → 3×H×W float32 [0,1]
        4. Normalize       — chuẩn hoá theo phân phối ImageNet

    Args:
        image_size  : kích thước output (mặc định 224)
        pad_fill    : giá trị pixel cho vùng padding (mặc định 0 = đen)
        mean        : ImageNet mean cho kênh R, G, B
        std         : ImageNet std  cho kênh R, G, B
        device      : 'cpu' hoặc 'cuda'
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        image_size: int = 224,
        pad_fill:   int = 0,
        mean: List[float] = None,
        std:  List[float] = None,
        device: str = "cpu",
    ):
        self.image_size = image_size
        self.pad_fill   = pad_fill
        self.device     = torch.device(device)

        # pad_square là custom step, thực hiện trước khi vào Compose
        self.transform = transforms.Compose([
            # Bước 1: Resize về image_size × image_size (ảnh đã vuông sau pad)
            transforms.Resize((image_size, image_size)),

            # Bước 2: H×W×3 uint8 [0,255] → 3×H×W float32 [0.0, 1.0]
            transforms.ToTensor(),

            # Bước 3: chuẩn hoá theo phân phối ImageNet
            transforms.Normalize(
                mean=mean or self.IMAGENET_MEAN,
                std =std  or self.IMAGENET_STD,
            ),
        ])

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _load_image(self, source: ImageInput) -> Image.Image:
        """
        Nhận file path hoặc PIL Image, luôn trả về PIL Image RGB.
        Dùng .convert("RGB") để xử lý ảnh RGBA (PNG có alpha)
        và ảnh grayscale (1 kênh).
        """
        if isinstance(source, (str, Path)):
            return Image.open(source).convert("RGB")
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        raise TypeError(f"Không hỗ trợ kiểu đầu vào: {type(source)}")

    def _apply_transform(self, pil_img: Image.Image) -> torch.Tensor:
        """
        Áp pad_square + transform pipeline và chuyển lên device.
        pad_square được gọi thủ công trước Compose vì cần truy cập self.pad_fill.
        """
        pil_img = pad_to_square(pil_img, fill=self.pad_fill)
        return self.transform(pil_img).to(self.device)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def process(self, source: ImageInput) -> torch.Tensor:
        """
        Xử lý 1 ảnh.

        Args:
            source: đường dẫn file (str / Path) hoặc PIL Image

        Returns:
            tensor shape (1, 3, 224, 224) — đã có batch dimension
        """
        pil_img = self._load_image(source)
        tensor  = self._apply_transform(pil_img)   # (3, 224, 224)
        return tensor.unsqueeze(0)                  # (1, 3, 224, 224)

    def process_batch(self, sources: List[ImageInput]) -> torch.Tensor:
        """
        Xử lý nhiều ảnh cùng lúc, stack thành 1 batch.

        Args:
            sources: list đường dẫn hoặc PIL Image

        Returns:
            tensor shape (N, 3, 224, 224)
        """
        tensors = [self._apply_transform(self._load_image(s)) for s in sources]
        return torch.stack(tensors).to(self.device)   # (N, 3, 224, 224)

    def get_transform(self) -> transforms.Compose:
        """
        Trả về transform pipeline (không gồm pad_square) để dùng với DataLoader.
        Khi dùng với DataLoader, gọi pad_to_square() trong __getitem__ trước khi
        truyền vào transform này.
        """
        return self.transform


# --------------------------------------------------------------------------- #
#  Quick test                                                                  #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import numpy as np

    processor = ImageProcessor(device="cpu")

    # Test 1: ảnh landscape (rộng hơn cao) — pad 2 bên trên/dưới
    wide_img = Image.fromarray(
        np.random.randint(0, 255, (300, 600, 3), dtype=np.uint8)
    )
    out = processor.process(wide_img)
    print(f"Landscape (300×600) → {out.shape}")     # (1, 3, 224, 224)

    # Test 2: ảnh portrait (cao hơn rộng) — pad 2 bên trái/phải
    tall_img = Image.fromarray(
        np.random.randint(0, 255, (600, 200, 3), dtype=np.uint8)
    )
    out2 = processor.process(tall_img)
    print(f"Portrait  (600×200) → {out2.shape}")    # (1, 3, 224, 224)

    # Test 3: ảnh vuông — không cần pad
    square_img = Image.fromarray(
        np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
    )
    out3 = processor.process(square_img)
    print(f"Square    (400×400) → {out3.shape}")    # (1, 3, 224, 224)

    print(f"dtype       : {out3.dtype}")             # torch.float32
    print(f"value range : [{out3.min():.3f}, {out3.max():.3f}]")

    # Test 4: batch
    batch = processor.process_batch([wide_img, tall_img, square_img])
    print(f"Batch 3 imgs: {batch.shape}")            # (3, 3, 224, 224)