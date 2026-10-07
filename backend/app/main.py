"""FastAPI inference API for the trained CXR mixture-of-experts model."""
from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import CHECKPOINT_PATH, CLASS_NAMES, CORS_ORIGINS, IMAGE_SIZE
from app.services.image_preprocessing import preprocess_upload

logger = logging.getLogger(__name__)

def load_model(checkpoint_path: Path | str = CHECKPOINT_PATH):
    """Load the exact training checkpoint; return None only when it is absent.

    Existing but corrupt, incomplete, or incompatible checkpoints deliberately raise,
    so the server cannot silently serve random initialized weights.
    """
    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.is_file():
        return None
    try:
        import torch
        from app.models.moe_system import CXRMoESystem

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:  # PyTorch versions predating weights_only
            payload = torch.load(checkpoint_path, map_location=device)
        if not isinstance(payload, dict) or "state_dict" not in payload:
            raise ValueError("expected the training payload dict containing 'state_dict'")
        if payload.get("class_names") != list(CLASS_NAMES):
            raise ValueError(
                f"checkpoint class order mismatch: expected {list(CLASS_NAMES)!r}, "
                f"found {payload.get('class_names')!r}"
            )
        state_dict = payload["state_dict"]
        if not isinstance(state_dict, dict) or not state_dict:
            raise ValueError("checkpoint 'state_dict' is empty or invalid")
        config = payload.get("config")
        if not isinstance(config, dict):
            raise ValueError("checkpoint is missing its training 'config'")
        image_size = config.get("image_size", IMAGE_SIZE)
        if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size <= 0:
            raise ValueError(f"invalid checkpoint image_size: {image_size!r}")

        # These are the same constructor arguments used by training/train.py.
        model = CXRMoESystem(num_classes=len(CLASS_NAMES))
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        return model, device, image_size
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load CXR checkpoint '{checkpoint_path}': "
            f"{type(exc).__name__}: {exc}"
        ) from exc

def _unpack_preprocessing_result(result: Any) -> tuple[Any, Any | None]:
    """Accept the current tensor-only API and tensor-plus-metadata results.

    When preprocessing returns a tuple, its first element is the model input;
    remaining element(s) are retained as metadata for the API response.
    """
    if isinstance(result, tuple):
        if not result:
            raise ValueError("preprocessing returned an empty tuple")
        metadata = result[1] if len(result) == 2 else (result[1:] or None)
        return result[0], metadata
    return result, None


def _prediction_response(output: Any) -> dict[str, Any]:
    """Validate the model's (logits, gates, aux) contract and build JSON-safe output."""
    import torch

    if not isinstance(output, (tuple, list)) or not output:
        raise ValueError("model forward must return (logits, [gating_weights, ...])")
    logits = output[0]
    if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (1, len(CLASS_NAMES)):
        shape = getattr(logits, "shape", None)
        raise ValueError(f"expected logits shape (1, {len(CLASS_NAMES)}), got {shape}")
    probabilities_tensor = torch.softmax(logits, dim=1)[0]
    if not bool(torch.isfinite(probabilities_tensor).all()):
        raise ValueError("model produced non-finite probabilities")
    probabilities = [float(x) for x in probabilities_tensor.detach().cpu().tolist()]
    predicted_index = int(probabilities_tensor.argmax().item())
    # Predictive entropy is a descriptive score-distribution proxy, not a statistical
    # confidence interval or a calibrated probability. Normalize to [0, 1].
    eps = torch.finfo(probabilities_tensor.dtype).tiny
    entropy = -(probabilities_tensor * probabilities_tensor.clamp_min(eps).log()).sum()
    normalized_entropy = float((entropy / torch.log(torch.tensor(
        len(CLASS_NAMES), dtype=entropy.dtype, device=entropy.device
    ))).detach().cpu().item())
    result: dict[str, Any] = {
        "classes": list(CLASS_NAMES),
        "probabilities": probabilities,
        "predicted_class": CLASS_NAMES[predicted_index],
        "uncertainty": {
            "method": "normalized_predictive_entropy",
            "score": normalized_entropy,
            "scale": "0-1 (higher means a more diffuse class score distribution)",
            "status": "exploratory_un_calibrated",
            "note": "This is not a statistical confidence interval or calibrated uncertainty estimate.",
        },
        "calibration": {
            "status": "unavailable",
            "method": None,
            "reason": "No validation logits/labels or fitted calibration artifact is installed. Supply an independent labeled validation set to fit and assess calibration.",
        },
    }
    # The current CXRMoESystem returns (fused_logits, gates, aux). Do not fabricate
    # gates for models whose forward output does not actually include them.
    if len(output) > 1 and isinstance(output[1], torch.Tensor):
        gates = output[1]
        if gates.ndim != 2 or gates.shape[0] != 1:
            raise ValueError(f"expected gating weights shape (1, experts), got {tuple(gates.shape)}")
        result["gating_weights"] = [float(x) for x in gates[0].detach().cpu().tolist()]
    return result

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        loaded = load_model(CHECKPOINT_PATH)
    except RuntimeError:
        # Fail loudly for a present but corrupt/incompatible checkpoint; never serve
        # randomly initialized weights as if inference were ready.
        logger.exception("CXR checkpoint is present but could not be loaded")
        raise
    if loaded is None:
        # Missing weights are an expected deployment state: serve health and 503 on
        # inference until a real training checkpoint is installed.
        logger.warning("CXR checkpoint not found at %s; inference remains unavailable", CHECKPOINT_PATH)
        app.state.model, app.state.device, app.state.image_size = None, None, IMAGE_SIZE
    else:
        app.state.model, app.state.device, app.state.image_size = loaded
    yield

app = FastAPI(title="CXR MoE System", lifespan=lifespan)

# Browser origins allowed to call this API (frontend served by nginx).
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health(request: Request):
    ready = getattr(request.app.state, "model", None) is not None
    return {"status": "ok", "inference_ready": ready}

@app.post("/preprocess")
async def preprocess(file: UploadFile = File(...)):
    """Validate and preprocess an image without running inference."""
    tensor, metadata = _unpack_preprocessing_result(await preprocess_upload(file))
    result = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "classes": CLASS_NAMES,
        "image_size": IMAGE_SIZE,
    }
    if metadata is not None:
        result["metadata"] = metadata
    return result

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    """Run inference for one uploaded chest X-ray using the trained checkpoint."""
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Inference unavailable: trained checkpoint not found at {CHECKPOINT_PATH}",
        )
    tensor, metadata = _unpack_preprocessing_result(
        await preprocess_upload(file, image_size=request.app.state.image_size)
    )
    device = request.app.state.device
    try:
        import torch
        with torch.inference_mode():
            output = model(tensor.to(device))
        result = _prediction_response(output)
        if metadata is not None:
            result["preprocessing_metadata"] = metadata
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {type(exc).__name__}: {exc}") from exc
