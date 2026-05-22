import logging
import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]


def get_logger(name="3D_Recon", log_dir=None, log_file="workflow.log"):
    if log_dir is None:
        log_dir = PROJECT_DIR / "results" / "chair_resnet_baseline" / "logs"
    else:
        log_dir = Path(log_dir)
        if not log_dir.is_absolute():
            log_dir = PROJECT_DIR / log_dir

    os.makedirs(log_dir, exist_ok=True)
    log_file_path = log_dir / log_file

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")

    if not any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename) == log_file_path
        for handler in logger.handlers
    ):
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if not any(type(handler) is logging.StreamHandler for handler in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger
