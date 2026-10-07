"""Inference safety and output contract tests."""
from pathlib import Path
import sys

# Make ``app`` importable when pytest is launched from either the backend or
# repository root (the application package lives directly under backend/).
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import pytest

torch = pytest.importorskip("torch")
from fastapi.testclient import TestClient
from app import main
from app import config
from app.config import CLASS_NAMES



def test_config_class_order_and_checkpoint_path_contract():
    assert CLASS_NAMES == ["COVID19", "NORMAL", "PNEUMONIA", "TURBERCULOSIS"]
    assert config.PROJECT_ROOT == Path(config.__file__).resolve().parent.parent.parent
    # Dockerfile uses /app as its working directory and compose mounts weights at
    # /app/weights; the same default points at <project-root>/weights on the host.
    if "CHECKPOINT_PATH" not in __import__("os").environ and "CXR_MOE_CHECKPOINT" not in __import__("os").environ:
        assert config.CHECKPOINT_PATH == config.WEIGHTS_PATH / "best_model.pt"
    if "WEIGHTS_PATH" not in __import__("os").environ and "CXR_MOE_WEIGHTS_DIR" not in __import__("os").environ:
        assert config.WEIGHTS_PATH == config.PROJECT_ROOT / "weights"

def test_missing_checkpoint_loader_returns_none(tmp_path):
    assert main.load_model(tmp_path / "missing" / "best_model.pt") is None


def test_missing_checkpoint_predict_returns_503(monkeypatch, tmp_path):
    missing = tmp_path / "best_model.pt"
    monkeypatch.setattr(main, "CHECKPOINT_PATH", missing)
    with TestClient(main.app) as client:
        response = client.post("/predict", files={"file": ("x.png", b"irrelevant", "image/png")})
    assert response.status_code == 503
    assert "checkpoint not found" in response.json()["detail"]


class FakeModel:
    def __call__(self, batch):
        assert tuple(batch.shape) == (1, 3, 224, 224)
        return (torch.tensor([[0., 0., 4., 0.]]),
                torch.tensor([[0.2, 0.3, 0.5]]), {"expert_logits": torch.zeros(1, 3, 4)})


def test_predict_unpacks_preprocessing_tensor_and_preserves_metadata(monkeypatch):
    metadata = {"source_width": 640, "source_height": 480}

    async def fake_preprocess(upload, image_size=224):
        return torch.zeros(1, 3, image_size, image_size), metadata

    monkeypatch.setattr(main, "preprocess_upload", fake_preprocess)
    with TestClient(main.app) as client:
        client.app.state.model = FakeModel()
        client.app.state.device = torch.device("cpu")
        client.app.state.image_size = 224
        response = client.post("/predict", files={"file": ("x.png", b"unused", "image/png")})
    assert response.status_code == 200, response.text
    assert response.json()["preprocessing_metadata"] == metadata


def test_predict_response_uses_exact_class_order_softmax_and_real_gates(monkeypatch):
    async def fake_preprocess(upload, image_size=224):
        return torch.zeros(1, 3, image_size, image_size)
    monkeypatch.setattr(main, "preprocess_upload", fake_preprocess)
    with TestClient(main.app) as client:
        client.app.state.model = FakeModel()
        client.app.state.device = torch.device("cpu")
        client.app.state.image_size = 224
        response = client.post("/predict", files={"file": ("x.png", b"unused", "image/png")})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["classes"] == CLASS_NAMES
    assert data["predicted_class"] == CLASS_NAMES[2]
    assert len(data["probabilities"]) == len(CLASS_NAMES)
    assert sum(data["probabilities"]) == pytest.approx(1.0)
    assert data["probabilities"][2] == pytest.approx(torch.softmax(torch.tensor([0., 0., 4., 0.]), 0)[2].item())
    assert data["gating_weights"] == pytest.approx([0.2, 0.3, 0.5])
    uncertainty = data["uncertainty"]
    assert uncertainty["method"] == "normalized_predictive_entropy"
    assert 0.0 <= uncertainty["score"] <= 1.0
    assert uncertainty["status"] == "exploratory_un_calibrated"
    assert "not a statistical confidence interval" in uncertainty["note"]
    assert data["calibration"]["status"] == "unavailable"
    assert data["calibration"]["method"] is None


def test_predict_omits_gates_when_forward_does_not_return_them():
    result = main._prediction_response((torch.zeros(1, 4),))
    assert result["classes"] == CLASS_NAMES
    assert result["probabilities"] == pytest.approx([0.25] * 4)
    assert "gating_weights" not in result
    assert result["uncertainty"]["score"] == pytest.approx(1.0)
    assert result["calibration"]["status"] == "unavailable"


def test_training_checkpoint_round_trip_strict_loader(tmp_path):
    from app.models.moe_system import CXRMoESystem
    path = tmp_path / "best_model.pt"
    trained_model = CXRMoESystem(num_classes=len(CLASS_NAMES), in_channels=3)
    torch.save({"state_dict": trained_model.state_dict(), "class_names": CLASS_NAMES,
                "config": {"image_size": 224}}, path)
    loaded, device, image_size = main.load_model(path)
    assert isinstance(loaded, CXRMoESystem)
    assert loaded.training is False
    assert device.type in ("cpu", "cuda")
    assert image_size == 224
    # Strict loading must reject a partial/incompatible state dict with a clear error.
    torch.save({"state_dict": {}, "class_names": CLASS_NAMES, "config": {"image_size": 224}}, path)
    with pytest.raises(RuntimeError, match="Unable to load CXR checkpoint"):
        main.load_model(path)
