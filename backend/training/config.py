from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class TrainConfig:
    data_root: Path = field(default_factory=lambda: Path("data"))
    output_dir: Path = field(default_factory=lambda: Path("weights"))
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 30
    num_workers: int = 0
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 5
    min_delta: float = 1e-4
    seed: int = 42
    amp: bool = True
    focal_gamma: float = 2.0
    focal_beta: float = 0.9999
    val_max_batches: int = 0

    def resolved(self) -> TrainConfig:
        return TrainConfig(
            data_root=Path(self.data_root).resolve(),
            output_dir=Path(self.output_dir).resolve(),
            image_size=int(self.image_size),
            batch_size=int(self.batch_size),
            epochs=int(self.epochs),
            num_workers=int(self.num_workers),
            learning_rate=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
            patience=int(self.patience),
            min_delta=float(self.min_delta),
            seed=int(self.seed),
            amp=bool(self.amp),
            focal_gamma=float(self.focal_gamma),
            focal_beta=float(self.focal_beta),
            val_max_batches=int(self.val_max_batches),
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["data_root"] = str(data["data_root"])
        data["output_dir"] = str(data["output_dir"])
        return data
