from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class DatasetSample:
    image_path: Path
    mask_path: Path


def _collect_image_files(path: Path) -> List[Path]:
    files = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
    files.sort()
    return files


def find_dataset_root() -> Path:
    """
    Supports dataset at either:
      - <repo>/dataset
      - <repo>/backend/dataset
    """
    backend_root = Path(__file__).resolve().parents[2]
    repo_root = backend_root.parent
    candidates = [repo_root / "dataset", backend_root / "dataset"]
    for candidate in candidates:
        if (candidate / "images").exists() and (candidate / "masks").exists():
            return candidate
    return candidates[0]


def load_dataset_samples(limit: int | None = None) -> List[DatasetSample]:
    root = find_dataset_root()
    image_dir = root / "images"
    mask_dir = root / "masks"
    if not image_dir.exists() or not mask_dir.exists():
        return []

    image_files = _collect_image_files(image_dir)
    mask_files = _collect_image_files(mask_dir)
    paired_count = min(len(image_files), len(mask_files))
    samples = [DatasetSample(image_files[i], mask_files[i]) for i in range(paired_count)]
    if limit is not None:
        return samples[: max(0, limit)]
    return samples


def read_grayscale_normalized(image_path: Path, size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image at: {image_path}")
    resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


class BratsSliceDataset(Dataset):
    def __init__(self, size: Tuple[int, int] = (128, 128), limit: int = 20):
        self.size = size
        self.samples = load_dataset_samples(limit=limit)
        if not self.samples:
            raise ValueError("Dataset not found or empty. Expected dataset/images and dataset/masks.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = read_grayscale_normalized(sample.image_path, self.size)
        mask = read_grayscale_normalized(sample.mask_path, self.size)
        mask = (mask >= 0.5).astype(np.float32)
        x = torch.from_numpy(image).unsqueeze(0)
        y = torch.from_numpy(mask).unsqueeze(0)
        return x, y
