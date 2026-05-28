from __future__ import annotations

import numpy as np
import torch


def marching_cubes(level: torch.Tensor, threshold: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU fallback compatible with the torchmcubes API used by TripoSR.

    The official TripoSR stack depends on the native torchmcubes extension. On
    Windows this extension often requires Visual Studio C++ build tools. This
    fallback keeps the core testable by delegating to scikit-image's marching
    cubes implementation.
    """

    try:
        from skimage import measure
    except Exception as exc:
        raise RuntimeError(
            "Install scikit-image or the native torchmcubes package for marching cubes."
        ) from exc

    device = level.device
    volume = level.detach().cpu().numpy().astype(np.float32)
    vertices, faces, _, _ = measure.marching_cubes(volume, level=float(threshold))
    vertices_tensor = torch.from_numpy(vertices.astype(np.float32)).to(device)
    faces_tensor = torch.from_numpy(faces.astype(np.int64)).to(device)
    return vertices_tensor, faces_tensor
