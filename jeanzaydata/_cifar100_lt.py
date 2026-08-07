"""CIFAR-100-LT array loading helpers."""

import os
import pickle
from pathlib import Path
from typing import Any
import numpy as np

_CIFAR100_NUM_CLASSES = 100
_CIFAR100_IMAGE_HW = 32
_CIFAR100_FILES = {
    "train": "train",
    "test": "test",
}
_CIFAR100_META_FILE = "meta"


def _normalize_cifar100_split(split: str) -> str:
    key = str(split).lower().strip()
    if key in {"validation", "valid", "val"}:
        key = "test"
    if key not in _CIFAR100_FILES:
        raise ValueError(
            f"Unsupported CIFAR-100 split {split!r}. Expected one of {sorted(_CIFAR100_FILES)}."
        )
    return key


def _cifar100_root(data_home: str | Path | None) -> Path:
    if data_home is not None:
        root = Path(data_home).expanduser()
    else:
        data_root = os.getenv("JEANZAY_DATA")
        if not data_root:
            raise RuntimeError(
                "CIFAR-100-LT raw data location is not configured. "
                "Set JEANZAY_DATA so the loader uses $JEANZAY_DATA/raw/cifar100_lt, "
                "or pass data_home explicitly. "
                "Run `python -m jeanzaydata init cifar100_lt` after exporting JEANZAY_DATA."
            )
        root = Path(data_root).expanduser() / "raw" / "cifar100_lt"
    if not root.is_dir():
        raise RuntimeError(
            f"CIFAR-100-LT root does not exist: {root}. "
            "Run `python -m jeanzaydata init cifar100_lt` to populate it."
        )
    return root


def _resolve_cifar100_file(root: Path, name: str) -> Path:
    candidates = [
        root / name,
        root / "cifar-100-python" / name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise RuntimeError(
        f"CIFAR-100 file {name!r} not found under {root}. Searched:\n{searched}"
    )


def _load_cifar100_pickle(path: Path) -> dict[bytes, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="bytes")


def _stack_cifar100_images(raw: np.ndarray) -> np.ndarray:
    """Convert (N, 3072) uint8 rows to (N, 32, 32, 3) HWC arrays."""
    if raw.ndim != 2 or raw.shape[1] != 3 * _CIFAR100_IMAGE_HW * _CIFAR100_IMAGE_HW:
        raise RuntimeError(
            f"Unexpected CIFAR-100 data shape {raw.shape}; expected (N, {3 * _CIFAR100_IMAGE_HW * _CIFAR100_IMAGE_HW})."
        )
    reshaped = raw.reshape(-1, 3, _CIFAR100_IMAGE_HW, _CIFAR100_IMAGE_HW)
    return np.transpose(reshaped, (0, 2, 3, 1))


def _load_cifar100_label_names(root: Path) -> list[str]:
    meta_path = _resolve_cifar100_file(root, _CIFAR100_META_FILE)
    meta = _load_cifar100_pickle(meta_path)
    raw_names = meta.get(b"fine_label_names", [])
    return [name.decode("utf-8") if isinstance(name, (bytes, bytearray)) else str(name) for name in raw_names]


def _long_tail_class_counts(
    n_max: int,
    *,
    imbalance_factor: float,
    n_classes: int,
) -> list[int]:
    if n_max <= 0:
        raise ValueError("n_max must be > 0.")
    if imbalance_factor <= 0:
        raise ValueError("imbalance_factor must be > 0.")
    if n_classes < 2:
        raise ValueError("n_classes must be >= 2 for the long-tail rule.")
    counts = []
    for c in range(n_classes):
        ratio = imbalance_factor ** (-c / (n_classes - 1))
        counts.append(max(1, int(np.floor(n_max * ratio))))
    return counts


def _select_long_tail_indices(
    labels: np.ndarray,
    *,
    counts_by_class: list[int],
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    selected: list[np.ndarray] = []
    for class_id, target in enumerate(counts_by_class):
        class_idx = np.where(labels == class_id)[0]
        if class_idx.size == 0:
            continue
        if class_idx.size <= target:
            selected.append(np.sort(class_idx))
            continue
        chosen = rng.choice(class_idx, size=target, replace=False)
        selected.append(np.sort(chosen))
    if not selected:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(selected).astype(np.int64)


def _apply_max_samples(
    indices: np.ndarray,
    *,
    max_samples: int | None,
    seed: int,
) -> np.ndarray:
    if max_samples is None or indices.size <= max_samples:
        return indices
    rng = np.random.default_rng(int(seed) + 1)
    chosen = rng.choice(indices, size=int(max_samples), replace=False)
    return np.sort(chosen).astype(np.int64)


def load_cifar100_lt_arrays(
    *,
    root: Path,
    split: str,
    imbalance_factor: float,
    max_samples: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[int]]:
    """Return ``(images_hwc_uint8, labels, selected_indices, label_names, class_counts)``."""
    split_key = _normalize_cifar100_split(split)
    file_path = _resolve_cifar100_file(root, _CIFAR100_FILES[split_key])
    raw = _load_cifar100_pickle(file_path)
    images = _stack_cifar100_images(np.asarray(raw[b"data"], dtype=np.uint8))
    labels = np.asarray(raw[b"fine_labels"], dtype=np.int64)
    if images.shape[0] != labels.shape[0]:
        raise RuntimeError(
            f"CIFAR-100 split {split_key!r} has mismatched data/label counts: "
            f"{images.shape[0]} images vs {labels.shape[0]} labels."
        )
    n_classes = int(labels.max()) + 1 if labels.size else _CIFAR100_NUM_CLASSES
    n_max = int(np.bincount(labels, minlength=n_classes).max())
    counts = _long_tail_class_counts(n_max, imbalance_factor=imbalance_factor, n_classes=n_classes)
    indices = _select_long_tail_indices(labels, counts_by_class=counts, seed=seed)
    indices = _apply_max_samples(indices, max_samples=max_samples, seed=seed)
    label_names = _load_cifar100_label_names(root)
    return images[indices], labels[indices], indices, label_names, counts
