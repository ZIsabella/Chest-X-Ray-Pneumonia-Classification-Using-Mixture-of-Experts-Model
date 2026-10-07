import io

import pytest
import torch
from fastapi import HTTPException
from PIL import Image

from app.services.image_preprocessing import preprocess_image_bytes


def encoded(mode, size, fmt, color):
    image = Image.new(mode, size, color)
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def test_png_rgb_preserves_rgb_resizes_and_normalizes():
    tensor = preprocess_image_bytes(encoded("RGB", (80, 40), "PNG", (255, 0, 0)), filename="x.png")
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32
    assert torch.allclose(tensor[:, 0], torch.ones_like(tensor[:, 0]), atol=1e-6)
    assert torch.allclose(tensor[:, 1:], -torch.ones_like(tensor[:, 1:]), atol=1e-6)


def test_jpeg_grayscale_is_batched_and_resized():
    tensor = preprocess_image_bytes(encoded("L", (31, 77), "JPEG", 0), filename="x.jpeg", image_size=64)
    assert tensor.shape == (1, 3, 64, 64)
    assert torch.allclose(tensor, -torch.ones_like(tensor), atol=1e-6)


def test_rejects_corrupt_file():
    with pytest.raises(HTTPException) as err:
        preprocess_image_bytes(b"not an image", filename="broken.png")
    assert err.value.status_code == 400


def test_rejects_mismatched_and_unsupported_format():
    png = encoded("L", (2, 3), "PNG", 100)
    with pytest.raises(HTTPException) as err:
        preprocess_image_bytes(png, filename="renamed.jpg")
    assert err.value.status_code == 415
    with pytest.raises(HTTPException) as err:
        preprocess_image_bytes(png, filename="x.gif")
    assert err.value.status_code == 415
