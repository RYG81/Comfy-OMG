"""
image_utils.py — Image ↔ base64 conversion for Ollama vision models.
"""
from __future__ import annotations

import base64
import hashlib
import io
from typing import Optional

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore


def tensor_to_base64(image_tensor, max_size: int = 1024) -> str:
    """
    Convert a ComfyUI IMAGE tensor (B,H,W,C float32 0-1) to a base64 JPEG string.
    Resizes if either dimension exceeds max_size (preserving aspect ratio).
    """
    if Image is None:
        raise ImportError("Pillow is required for image processing")

    # Handle batch dimension — take first image
    if len(image_tensor.shape) == 4:
        image_tensor = image_tensor[0]

    # Convert to numpy uint8
    img_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)

    # Resize if too large
    w, h = pil_img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

    return pil_to_base64(pil_img)


def image_tensor_hash(image_tensor) -> str:
    """Return a stable hash for ComfyUI IMAGE tensor pixel data."""
    if hasattr(image_tensor, "detach"):
        image_tensor = image_tensor.detach()
    if hasattr(image_tensor, "cpu"):
        image_tensor = image_tensor.cpu()
    if hasattr(image_tensor, "numpy"):
        image_tensor = image_tensor.numpy()

    arr = np.ascontiguousarray(image_tensor)
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode("utf-8"))
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(arr.tobytes())
    return digest.hexdigest()


def pil_to_base64(pil_img, quality: int = 85) -> str:
    """Convert a PIL Image to a base64-encoded JPEG string."""
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
