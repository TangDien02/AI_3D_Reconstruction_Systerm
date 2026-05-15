import os
import sys
from dotenv import load_dotenv

# Đảm bảo Python nhận diện được thư mục src
sys.path.append(os.path.dirname(__file__))

from src.utils.logger import get_logger
from src.pipeline.baseline_runner import BaselinePipeline
from src.utils.cloud import S3Manager

# Load cấu hình
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=env_path)

logger = get_logger("MainWorkflow")

def main():
 
    template_db = {
        "chair": "data/templates/chair_avg.obj",
        "table": "data/templates/table_avg.obj"
    }

    # 2. Khởi tạo Pipeline
    pipeline = BaselinePipeline(template_db=template_db)

    # 3. Giả lập Metadata từ tập Pre-Data
    # Bạn có thể đọc JSON thực tế ở đây: json.load(open("data/metadata.json"))
    pre_data_metadata = {
        "img_001": "chair",
        "img_002": "table"
    }

    # 4. Danh sách các file cần chạy (Test cases)
    test_cases = [
        # Thay vì chỉ lưu file ở máy, ta khai báo đường dẫn nó nằm trên S3 (s3_key)
        # và đường dẫn ta muốn lưu tạm ở máy (local_path)
        {
            "id": "img_001",
            "s3_img_key": "pix3d/images/chair/chair_001.jpg", 
            "s3_gt_key": "pix3d/models/chair/chair_001.obj",
            "local_img": "data/raw/chair1.jpg", 
            "local_gt": "data/raw/chair1_gt.obj"
        },
    ]

    logger.info(f"Tổng số ảnh cần chạy: {len(test_cases)}")
    
    s3_manager = S3Manager() # Khởi tạo S3 Manager
    
    # 5. Đảm bảo Bucket tồn tại trên Docker/AWS (tránh xung đột lỗi 404)
    try:
        s3_manager.s3_client.head_bucket(Bucket=s3_manager.bucket_name)
        logger.info(f"Bucket {s3_manager.bucket_name} đã sẵn sàng.")
    except:
        logger.info(f"Bucket {s3_manager.bucket_name} chưa tồn tại, tiến hành tạo mới trên giả lập...")
        try:
            s3_manager.s3_client.create_bucket(
                Bucket=s3_manager.bucket_name,
                CreateBucketConfiguration={'LocationConstraint': 'ap-southeast-1'}
            )
        except Exception as e:
            logger.warning(f"Không thể tạo bucket (có thể đang dùng mock thật): {e}")

    # 6. Vòng lặp Workflow
    for item in test_cases:
        # --- BƯỚC A: Kéo data từ S3 về máy ---
        logger.info(f"Tiến hành kéo data từ S3 cho ảnh {item['id']}...")
        s3_manager.download_file(item["s3_img_key"], item["local_img"])
        s3_manager.download_file(item["s3_gt_key"], item["local_gt"])
        
        # --- BƯỚC B: Chạy thuật toán 3D ---
        cd_fscore = pipeline.run_single(
            test_image_path=item["local_img"],
            gt_mesh_path=item["local_gt"],
            image_id=item["id"],
            pre_data_metadata=pre_data_metadata
        )
        
        if cd_fscore:
            logger.info(f"✅ Hoàn thành {item['id']}")
        else:
            logger.warning(f"❌ Lỗi xử lý {item['id']}")

    logger.info("Hoàn tất Workflow!")

if __name__ == "__main__":
    main()
