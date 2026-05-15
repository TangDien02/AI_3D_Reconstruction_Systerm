import numpy as np
import trimesh
from scipy.spatial import KDTree
from src.utils.logger import get_logger

logger = get_logger("Evaluator3D")

def sample_points_from_mesh(mesh_path, num_points=2048):
    try:
        mesh = trimesh.load(mesh_path, force='mesh')
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
        return points
    except Exception as e:
        logger.error(f"Lỗi load model 3D {mesh_path}: {e}")
        raise e

def compute_metrics(pred_mesh_path, gt_mesh_path, threshold=0.01):
    """
    Tính Chamfer Distance và F-score giữa 2 Point Clouds.
    """
    pred_pc = sample_points_from_mesh(pred_mesh_path)
    gt_pc = sample_points_from_mesh(gt_mesh_path)
    
    tree_pred = KDTree(pred_pc)
    tree_gt = KDTree(gt_pc)
    
    # Tính khoảng cách L2
    dist_gt_to_pred, _ = tree_pred.query(gt_pc)
    dist_pred_to_gt, _ = tree_gt.query(pred_pc)
    
    # Chamfer Distance
    cd = np.mean(dist_gt_to_pred**2) + np.mean(dist_pred_to_gt**2)
    
    # F-score
    precision = np.mean(dist_pred_to_gt < threshold)
    recall = np.mean(dist_gt_to_pred < threshold)
    f_score = 0.0
    if precision + recall > 0:
        f_score = 2 * (precision * recall) / (precision + recall)
        
    return cd, f_score
