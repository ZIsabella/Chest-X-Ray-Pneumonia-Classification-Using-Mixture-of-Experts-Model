"""Safe uploaded-image decoding and preprocessing for the CXR model.

Matches training/dataset.py: RGB, square resize, ToTensor, Normalize(0.5, 0.5) per channel.
"""
from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import Final

import numpy as np
import torch
from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app.config import IMAGE_MEAN, IMAGE_SIZE, IMAGE_STD, MAX_UPLOAD_BYTES

# Keep this allowlist aligned with training/dataset.py. Format is checked from
# decoded bytes, not merely trusted from the client-supplied filename/MIME type.
SUPPORTED_EXTENSIONS: Final = {
    ".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".bmp": "BMP",
    ".tif": "TIFF", ".tiff": "TIFF", ".webp": "WEBP",
}


def preprocess_image_bytes(
    content: bytes, *, filename: str = "image.png", image_size: int = IMAGE_SIZE,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> torch.Tensor:
    """Decode supported image bytes and return normalized float tensor [1,3,H,W].

    Pixel values follow the training transform: RGB, square bilinear resize,
    [0,1] conversion, then (x - 0.5) / 0.5 per channel (approximately [-1,1]).
    """
    if not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("image_size must be a positive integer")
    if not isinstance(content, (bytes, bytearray)) or not content:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Image exceeds {max_bytes} byte limit")
    suffix = Path(filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Unsupported image extension")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as image:
                if image.format not in {"JPEG", "PNG", "BMP", "TIFF", "WEBP"}:
                    raise HTTPException(status_code=415, detail=f"Unsupported image format: {image.format}")
                image.verify()
            # verify() checks the file structure but invalidates that image object;
            # reopen and fully decode before accepting the upload.
            with Image.open(io.BytesIO(content)) as image:
                if image.width * image.height > Image.MAX_IMAGE_PIXELS:
                    raise HTTPException(status_code=413, detail="Image dimensions exceed safety limit")
                image.load()
                rgb = image.convert("RGB")
                rgb = rgb.resize((image_size, image_size), resample=Image.Resampling.BILINEAR)
                array = np.asarray(rgb, dtype=np.float32) / 255.0
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image") from exc

    array = (array - IMAGE_MEAN) / IMAGE_STD
    # [H,W,C] -> [C,H,W] -> [1,C,H,W]. Ensure writable contiguous storage for torch.
    array = np.ascontiguousarray(array.transpose(2, 0, 1))
    tensor = torch.from_numpy(array).unsqueeze(0)
    return tensor


async def preprocess_upload(
    upload: UploadFile, *, image_size: int = IMAGE_SIZE, max_bytes: int = MAX_UPLOAD_BYTES
) -> torch.Tensor:
    """Read a FastAPI upload with a hard byte cap and preprocess it."""
    try:
        content = await upload.read(max_bytes + 1)
    finally:
        await upload.close()
    return preprocess_image_bytes(content, filename=upload.filename or "", image_size=image_size, max_bytes=max_bytes)
