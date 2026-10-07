"""Validation and optional PyTorch input pipeline for the CXR dataset."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image

from app.config import CLASS_NAMES as _CLASS_NAMES

# Keep the training label-index contract tied to the inference API contract.
CLASS_NAMES: Tuple[str, ...] = tuple(_CLASS_NAMES)
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
SPLITS = ("train", "validation", "test")

try:
    import torch
    from torch.utils.data import DataLoader, Dataset as TorchDataset
    from torchvision import transforms
    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # Missing torch OR an incompatible/broken torchvision install.
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment,misc]
    TorchDataset = object  # type: ignore[assignment,misc]
    transforms = None  # type: ignore[assignment]


@dataclass
class SplitReport:
    split: str
    exists: bool = False
    counts: Dict[str, int] = field(default_factory=dict)
    missing_classes: List[str] = field(default_factory=list)
    extra_dirs: List[str] = field(default_factory=list)
    stray_files: List[str] = field(default_factory=list)
    unreadable: List[str] = field(default_factory=list)
    total: int = 0


def _list_images(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def validate_split(data_root: Path | str, split: str, check_readable: bool = False) -> SplitReport:
    """Inspect data_root/split/class/images. Unrecognized top-level entries are reported."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
    split_dir = Path(data_root) / split
    report = SplitReport(split=split, counts={name: 0 for name in CLASS_NAMES})
    if not split_dir.is_dir():
        return report
    report.exists = True
    for cls in CLASS_NAMES:
        cls_dir = split_dir / cls
        if not cls_dir.is_dir():
            report.missing_classes.append(cls)
            continue
        images = _list_images(cls_dir)
        report.counts[cls] = len(images)
        report.total += len(images)
        if check_readable:
            for img in images:
                try:
                    with Image.open(img) as im:
                        im.verify()
                except Exception:
                    report.unreadable.append(str(img.relative_to(split_dir)))
    for entry in split_dir.iterdir():
        if entry.is_dir() and entry.name not in CLASS_NAMES:
            report.extra_dirs.append(entry.name)
        elif entry.is_file() and entry.name != ".gitkeep":
            report.stray_files.append(entry.name)
    return report


def validate_dataset(data_root: Path | str, check_readable: bool = False,
                     require_all: bool = False) -> Dict[str, SplitReport]:
    """Return all split reports. require_all raises if any split/class folder is absent."""
    root = Path(data_root)
    reports = {s: validate_split(root, s, check_readable) for s in SPLITS}
    if require_all:
        problems = [f"{s}: missing split" if not r.exists else
                    f"{s}: missing class folders {r.missing_classes}"
                    for s, r in reports.items() if not r.exists or r.missing_classes]
        if problems:
            raise ValueError("Incomplete dataset: " + "; ".join(problems))
    return reports


def print_dataset_report(reports: Dict[str, SplitReport]) -> None:
    header = f"{'split':<12}" + "".join(f"{c:>16}" for c in CLASS_NAMES) + f"{'total':>9}"
    print(header)
    print("-" * len(header))
    for split in SPLITS:
        r = reports[split]
        if not r.exists:
            print(f"{split:<12}  -- MISSING --")
            continue
        print(f"{split:<12}" + "".join(f"{r.counts[c]:>16}" for c in CLASS_NAMES) + f"{r.total:>9}")
        if r.missing_classes:
            print(f"    missing class folders: {r.missing_classes}")
        if r.extra_dirs:
            print(f"    unexpected folders: {r.extra_dirs}")
        if r.stray_files:
            print(f"    stray files: {r.stray_files[:10]}")
        if r.unreadable:
            print(f"    unreadable images: {len(r.unreadable)} e.g. {r.unreadable[:3]}")


def dataset_is_valid(reports: Dict[str, SplitReport], splits: Tuple[str, ...] = SPLITS) -> bool:
    return all(reports[s].exists and reports[s].total > 0 and
               not reports[s].missing_classes and not reports[s].unreadable for s in splits)


def build_transform(image_size: int = 224, train: bool = False):
    if not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("image_size must be a positive integer")
    if not TORCH_AVAILABLE:
        raise RuntimeError(f"PyTorch image transforms unavailable: {TORCH_IMPORT_ERROR}")
    pipeline = [transforms.Lambda(lambda image: image.convert("RGB")), transforms.Resize((image_size, image_size))]
    if train:
        pipeline.extend([transforms.RandomHorizontalFlip(p=0.5),
                         transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05))])
    pipeline.extend([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    return transforms.Compose(pipeline)


if TORCH_AVAILABLE:
    class CXRClassDataset(TorchDataset):
        """PyTorch dataset with labels fixed to CLASS_NAMES order."""
        def __init__(self, data_root: Path | str, split: str = "train", image_size: int = 224,
                     train: Optional[bool] = None, transform: Optional[Callable] = None):
            if split not in SPLITS:
                raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
            if image_size <= 0:
                raise ValueError("image_size must be positive")
            self.root, self.split = Path(data_root), split
            self.samples: List[Tuple[Path, int]] = []
            for idx, cls in enumerate(CLASS_NAMES):
                self.samples.extend((p, idx) for p in _list_images(self.root / split / cls))
            if not self.samples:
                raise RuntimeError(f"No supported images found in {self.root / split}")
            if train is None:
                train = split == "train"
            self.transform = transform if transform is not None else build_transform(image_size, train=train)

        def __len__(self):
            return len(self.samples)

        @property
        def class_to_idx(self) -> Dict[str, int]:
            return {name: i for i, name in enumerate(CLASS_NAMES)}

        def __getitem__(self, i: int):
            path, label = self.samples[i]
            with Image.open(path) as im:
                image = im.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            return image, label
else:
    class CXRClassDataset:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError(f"PyTorch Dataset unavailable: {TORCH_IMPORT_ERROR}")


def make_dataloader(data_root: Path | str, split: str = "train", image_size: int = 224,
                    batch_size: int = 32, num_workers: int = 0, shuffle: Optional[bool] = None,
                    train: Optional[bool] = None):
    if not TORCH_AVAILABLE:
        raise RuntimeError(f"PyTorch DataLoader unavailable: {TORCH_IMPORT_ERROR}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if shuffle is None:
        shuffle = split == "train"
    ds = CXRClassDataset(data_root, split=split, image_size=image_size, train=train)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available())
