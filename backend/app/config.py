# backend/app/config.py
from pathlib import Path
import os

CLASS_NAMES = ["COVID19", "NORMAL", "PNEUMONIA", "TURBERCULOSIS"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

WEIGHTS_PATH = Path(
    os.environ.get("WEIGHTS_PATH", os.environ.get("CXR_MOE_WEIGHTS_DIR", PROJECT_ROOT / "weights"))
).expanduser()

CHECKPOINT_PATH = Path(
    os.environ.get(
        "CHECKPOINT_PATH",
        os.environ.get("CXR_MOE_CHECKPOINT", WEIGHTS_PATH / "best_model.pt"),
    )
).expanduser()

DATA_PATH = Path(os.environ.get("CXR_MOE_DATA_DIR", PROJECT_ROOT / "data"))

CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "CORS_ORIGINS",
        "http://localhost:13000,http://127.0.0.1:13000,http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]

# Image Preprocessing & Validation
IMAGE_SIZE = 224
IMAGE_MEAN = 0.5
IMAGE_STD = 0.5
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

ALLOWED_IMAGE_TYPES = {"JPEG", "PNG", "BMP", "TIFF", "WEBP"}
ALLOWED_IMAGE_EXTENSIONS = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".bmp": "BMP",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".webp": "WEBP",
}
ALLOWED_IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/tiff",
    "image/webp",
}
