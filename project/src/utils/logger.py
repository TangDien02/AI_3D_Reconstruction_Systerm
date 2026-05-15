import os
import logging

def get_logger(name="3D_Recon"):
    # Lưu file log tại project/results/logs/workflow.log
    LOG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'logs')
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file_path = os.path.join(LOG_DIR, 'workflow.log')
    
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s')
        
        fh = logging.FileHandler(log_file_path, encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger
