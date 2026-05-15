from ultralytics import YOLO
from src.utils.logger import get_logger

logger = get_logger("ImagePreprocessor")

class ImagePreprocessor:
    def __init__(self):
        try:
            logger.info("Khởi tạo mô hình YOLOv8-seg (Fallback)...")
            self.yolo_model = YOLO('yolov8n-seg.pt')
        except Exception as e:
            logger.warning(f"Không thể khởi tạo YOLO: {e}")
            self.yolo_model = None

    def get_class_from_pre_data(self, image_id, metadata_dict):
        """ Lấy class từ pre-data (như file metadata JSON) """
        if metadata_dict and image_id in metadata_dict:
            return metadata_dict[image_id]
        return None

    def process(self, image_path, pre_data_class=None):
        """ Logic kết hợp: Ưu tiên Pre-Data, Fallback dùng YOLO """
        if pre_data_class:
            logger.info(f"Sử dụng class từ Pre-Data: {pre_data_class}")
            return pre_data_class

        if self.yolo_model:
            logger.info(f"Chạy YOLO Infer cho: {image_path}")
            results = self.yolo_model(image_path, verbose=False)
            if len(results[0].boxes) > 0:
                box = results[0].boxes[0]
                cls_id = int(box.cls[0])
                return self.yolo_model.names[cls_id]
                
        return None
