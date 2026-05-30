from src.preprocessing.image_processor import ImagePreprocessor
from src.metrics.evaluator import compute_metrics
from src.utils.logger import get_logger

logger = get_logger("BaselinePipeline")

# Legacy template-matching baseline kept for comparison. New image-to-3D
# orchestration should use src.pipeline.sequential_3d_pipeline.

class BaselinePipeline:
    def __init__(self, template_db):
        self.preprocessor = ImagePreprocessor()
        self.template_db = template_db

    def run_single(self, test_image_path, gt_mesh_path, image_id=None, pre_data_metadata=None):
        """
        Luồng chính cho 1 điểm dữ liệu (1 ảnh đầu vào)
        """
        logger.info(f"--- Đang xử lý: {test_image_path} ---")
        
        # 1. Tiền xử lý (Lấy class)
        pre_data_class = self.preprocessor.get_class_from_pre_data(image_id, pre_data_metadata)
        pred_class = self.preprocessor.process(test_image_path, pre_data_class=pre_data_class)
        
        if not pred_class:
            logger.warning("Không lấy được class. Bỏ qua.")
            return None
            
        logger.info(f"Class: {pred_class}")
        
        # 2. Truy xuất model Template 3D
        if pred_class not in self.template_db:
            logger.warning(f"Không có template cho '{pred_class}'. Bỏ qua.")
            return None
            
        pred_mesh_path = self.template_db[pred_class]
        logger.info(f"Template Model: {pred_mesh_path}")
        
        # 3. Đánh giá (Evaluate Metrics)
        try:
            cd, f_score = compute_metrics(pred_mesh_path, gt_mesh_path)
            logger.info(f"Kết quả -> Chamfer Distance: {cd:.6f} | F-score: {f_score:.4f}")
            return cd, f_score
        except Exception as e:
            logger.error(f"Đánh giá 3D thất bại: {e}")
            return None
