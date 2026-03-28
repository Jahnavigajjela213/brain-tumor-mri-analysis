from __future__ import annotations

import numpy as np
import torch


def image_to_tensor(gray_norm_2d: np.ndarray) -> torch.Tensor:
    """
    Input: (H, W) float32 in [0,1]
    Output: (1, 1, H, W) float32 tensor on CPU
    """
    if gray_norm_2d.ndim != 2:
        raise ValueError("Expected 2D grayscale image.")
    t = torch.from_numpy(gray_norm_2d).float().unsqueeze(0).unsqueeze(0)
    return t


def to_cpu_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy()

