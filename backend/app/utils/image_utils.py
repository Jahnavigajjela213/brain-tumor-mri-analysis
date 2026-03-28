from __future__ import annotations

import io
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

from app.utils.errors import bad_request


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    data = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise bad_request("Invalid image upload. Please upload a valid image file (png/jpg).")
    return img


def preprocess_mri(img_bgr: np.ndarray, size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    img_resized = cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # Simple explicit augmentation as requested
    # Random horizontal flip for robust segmentation
    if np.random.rand() > 0.8: # Apply in 20% of cases for demo
        gray = cv2.flip(gray, 1)
        
    norm = gray.astype(np.float32) / 255.0
    return norm


def apply_augmentation_pipeline(img_norm: np.ndarray) -> np.ndarray:
    """
    Applies standard augmentation transforms for medical imaging:
    - Random Brightness
    - Gaussian Noise
    """
    # Random brightness shift
    shift = (np.random.rand() - 0.5) * 0.1
    img_aug = np.clip(img_norm + shift, 0, 1)
    
    # Gaussian Noise
    noise = np.random.normal(0, 0.01, img_aug.shape)
    img_aug = np.clip(img_aug + noise, 0, 1)
    
    return img_aug.astype(np.float32)


def mask_to_png_bytes(mask_01: np.ndarray) -> bytes:
    mask_u8 = (np.clip(mask_01, 0.0, 1.0) * 255.0).astype(np.uint8)
    pil_img = Image.fromarray(mask_u8, mode="L")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

