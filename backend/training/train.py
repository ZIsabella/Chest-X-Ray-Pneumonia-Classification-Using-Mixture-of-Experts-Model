#!/usr/bin/env python3
"""Train the CXR MoE classifier. Run from project root: python -m backend.training.train"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .config import TrainConfig
from .dataset import CLASS_NAMES, make_dataloader


def _torch_modules():
    try:
        import torch
        from ..app.models.losses import ClassBalancedFocalLoss
        from ..app.models.moe_system import CXRMoESystem
        return torch, ClassBalancedFocalLoss, CXRMoESystem
    except Exception as exc:
        raise RuntimeError(f"PyTorch/model imports failed: {type(exc).__name__}: {exc}") from exc


def seed_everything(seed: int, torch) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, TypeError):
        pass


def _run_epoch(model, loader, criterion, device, torch, optimizer=None, scaler=None,
               amp: bool = False, max_batches: int = 0) -> Tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total = 0
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=(device.type == "cuda"))
        labels = labels.to(device, dtype=torch.long, non_blocking=(device.type == "cuda"))
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=bool(amp and device.type == "cuda")):
                logits, _gates, _aux = model(images)
                loss = criterion(logits, labels)
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        n = int(labels.shape[0])
        total_loss += float(loss.detach().item()) * n
        total += n
        predicted = logits.detach().argmax(dim=1).cpu().numpy()
        truth = labels.detach().cpu().numpy()
        np.add.at(matrix, (truth, predicted), 1)
    if total == 0:
        raise RuntimeError("DataLoader yielded no batches; check the selected dataset split")
    return total_loss / total, matrix, np.asarray([total])


def metrics_from_confusion(matrix: np.ndarray) -> Dict[str, object]:
    tp = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    sensitivity = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    f1 = np.divide(2 * precision * sensitivity, precision + sensitivity,
                   out=np.zeros_like(tp), where=(precision + sensitivity) > 0)
    return {"accuracy": float(tp.sum() / max(float(matrix.sum()), 1.0)),
            "macro_f1": float(f1.mean()), "macro_precision": float(precision.mean()),
            "macro_sensitivity": float(sensitivity.mean()),
            "per_class_sensitivity": {name: float(sensitivity[i]) for i, name in enumerate(CLASS_NAMES)},
            "support": {name: int(support[i]) for i, name in enumerate(CLASS_NAMES)}}


def parse_args(argv=None):
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Train the four-class CXR MoE model")
    parser.add_argument("--data-root", type=Path, default=defaults.data_root)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--image-size", type=int, default=defaults.image_size)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--min-delta", type=float, default=defaults.min_delta)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=defaults.amp,
                        help="Use CUDA automatic mixed precision (default enabled; CPU always full precision)")
    parser.add_argument("--focal-gamma", type=float, default=defaults.focal_gamma)
    parser.add_argument("--focal-beta", type=float, default=defaults.focal_beta)
    parser.add_argument("--val-max-batches", type=int, default=defaults.val_max_batches,
                        help="Limit validation batches; 0 evaluates the full validation split")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if (args.image_size <= 0 or args.batch_size <= 0 or args.epochs <= 0 or args.num_workers < 0
            or args.learning_rate <= 0 or args.weight_decay < 0 or args.patience <= 0
            or args.min_delta < 0 or args.focal_gamma < 0 or not 0 <= args.focal_beta < 1
            or args.val_max_batches < 0):
        raise SystemExit("Invalid option: sizes/epochs/lr/patience must be positive; workers, delta, decay and batch limit non-negative; beta in [0,1)")
    torch, FocalLoss, Model = _torch_modules()
    cfg = TrainConfig(data_root=args.data_root, output_dir=args.output_dir,
        image_size=args.image_size, batch_size=args.batch_size, epochs=args.epochs,
        num_workers=args.num_workers, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, patience=args.patience, min_delta=args.min_delta,
        seed=args.seed, amp=args.amp, focal_gamma=args.focal_gamma,
        focal_beta=args.focal_beta, val_max_batches=args.val_max_batches).resolved()
    seed_everything(cfg.seed, torch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; AMP: {bool(cfg.amp and device.type == 'cuda')}; data: {cfg.data_root}")
    train_loader = make_dataloader(cfg.data_root, "train", cfg.image_size, cfg.batch_size,
                                   cfg.num_workers, shuffle=True, train=True)
    val_loader = make_dataloader(cfg.data_root, "validation", cfg.image_size, cfg.batch_size,
                                 cfg.num_workers, shuffle=False, train=False)
    counts = [0] * len(CLASS_NAMES)
    for _path, label in train_loader.dataset.samples:
        counts[int(label)] += 1
    if any(n <= 0 for n in counts):
        raise ValueError(f"Every training class needs at least one image for class-balanced loss; counts={counts}")
    print("Training counts: " + json.dumps(dict(zip(CLASS_NAMES, counts))))
    model = Model(num_classes=len(CLASS_NAMES)).to(device)
    criterion = FocalLoss(class_counts=counts, beta=cfg.focal_beta, gamma=cfg.focal_gamma).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.amp and device.type == "cuda"))
    except (AttributeError, TypeError):  # compatibility with older torch builds
        scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.amp and device.type == "cuda"))

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = cfg.output_dir / "best_model.pt"
    csv_path = cfg.output_dir / "training_history.csv"
    fields = ["epoch", "train_loss", "val_loss", "val_macro_f1", "val_accuracy", "learning_rate"] + [f"sensitivity_{n}" for n in CLASS_NAMES]
    best_f1, best_epoch, stale = -1.0, 0, 0
    with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, cfg.epochs + 1):
            train_loss, train_matrix, _ = _run_epoch(model, train_loader, criterion, device, torch,
                                                    optimizer, scaler, cfg.amp)
            train_accuracy = metrics_from_confusion(train_matrix)["accuracy"]
            val_loss, val_matrix, _ = _run_epoch(model, val_loader, criterion, device, torch,
                amp=cfg.amp, max_batches=cfg.val_max_batches)
            val_metrics = metrics_from_confusion(val_matrix)
            scheduler.step(val_loss)
            row = {"epoch": epoch, "train_loss": f"{train_loss:.6f}", "val_loss": f"{val_loss:.6f}",
                   "val_macro_f1": f"{val_metrics['macro_f1']:.6f}",
                   "val_accuracy": f"{val_metrics['accuracy']:.6f}",
                   "learning_rate": f"{optimizer.param_groups[0]['lr']:.8g}"}
            row.update({f"sensitivity_{k}": f"{v:.6f}" for k, v in val_metrics["per_class_sensitivity"].items()})
            writer.writerow(row); csv_file.flush(); sys.stdout.flush()
            sens = val_metrics["per_class_sensitivity"]
            print(f"Epoch {epoch}/{cfg.epochs} | train_loss={train_loss:.4f} | "
                  f"train_accuracy={train_accuracy:.4f} | val_loss={val_loss:.4f} | "
                  f"val_accuracy={val_metrics['accuracy']:.4f} | "
                  f"val_macro_f1={val_metrics['macro_f1']:.4f} | sensitivity={sens}", flush=True)
            if val_metrics["macro_f1"] > best_f1 + cfg.min_delta:
                best_f1, best_epoch, stale = val_metrics["macro_f1"], epoch, 0
                payload = {"state_dict": model.state_dict(), "class_names": list(CLASS_NAMES),
                    "config": cfg.to_dict(), "class_counts": dict(zip(CLASS_NAMES, counts)),
                    "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
                    "best_val_confusion_matrix": val_matrix.tolist()}
                tmp_path = checkpoint_path.with_suffix(".tmp")
                torch.save(payload, tmp_path)
                tmp_path.replace(checkpoint_path)
                print(f"  Saved new best checkpoint: {checkpoint_path}", flush=True)
            else:
                stale += 1
            if stale >= cfg.patience:
                print(f"Early stopping: no macro-F1 improvement for {cfg.patience} epochs.")
                break
    print(f"Best validation macro-F1={best_f1:.6f} at epoch {best_epoch}; checkpoint={checkpoint_path}; history={csv_path}")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"TRAINING FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
