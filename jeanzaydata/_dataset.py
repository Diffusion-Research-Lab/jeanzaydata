"""Real-dataset registry and public loading helpers."""

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import time
from typing import Any, Callable
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from ._cifar100_lt import _cifar100_root, _normalize_cifar100_split, load_cifar100_lt_arrays
from ._imagenet_lt import _imagenet_lt_record, _normalize_imagenet_lt_split, _parse_imagenet_lt_split_file, _resolve_imagenet_lt_annotation, _resolve_imagenet_lt_image_path, _resolve_imagenet_root, _select_imagenet_lt_records
from ._lvis import _find_lvis_annotation, _find_lvis_image_dirs, _lvis_base_roots, _lvis_candidate_images, _lvis_category_ids_by_image, _lvis_category_maps, _lvis_record, _lvis_search_roots, _load_lvis_json, _normalize_lvis_split, _resolve_lvis_image_path, _select_lvis_images


@dataclass(frozen=True)
class DatasetPayload:
    """Container for loaded data plus optional per-sample metadata."""

    data: pd.DataFrame | torch.Tensor | np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


DatasetLoader = Callable[..., pd.DataFrame | torch.Tensor | np.ndarray | DatasetPayload]
_IMAGE_CACHE_VERSION = 1


def _resampling() -> Any:
    return getattr(getattr(Image, "Resampling", Image), "BILINEAR")


def array_uint8_to_resized_tensor(array: np.ndarray, image_size: int) -> torch.Tensor:
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 uint8 image array, got shape {array.shape}.")
    image = Image.fromarray(array.astype(np.uint8, copy=False), mode="RGB")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), _resampling())
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(pixels).permute(2, 0, 1).contiguous()


def read_rgb_resized(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (image_size, image_size):
            image = image.resize((image_size, image_size), _resampling())
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(pixels).permute(2, 0, 1).contiguous()


@dataclass(frozen=True)
class DatasetEntry:
    """Describe one dataset exposed through the public dataset API."""

    name: str
    dataset_type: str
    description: str
    tail_index_alpha: Any = None
    split_mode: str = "random"
    standardize_default: bool = True
    dim: int | tuple[int, ...] | None = None
    n_samples: int | None = None
    loader: DatasetLoader | None = None

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tail_index_alpha": self.tail_index_alpha,
            "description": self.description,
            "split_mode": self.split_mode,
            "dataset_type": self.dataset_type,
            "dim": self.dim,
            "n_samples": self.n_samples,
        }


def _metadata_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _metadata_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metadata_value(val) for val in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (torch.device, torch.dtype)):
        return str(value)
    return value


def _resolve_real_data_home() -> Path:
    candidates: list[Path] = []
    for variable in ("JEANZAY_DATA", "JEANZAY_DATA_HOME"):
        value = os.getenv(variable)
        if value:
            candidates.append(Path(value).expanduser())
    work_root = os.getenv("WORK")
    if work_root:
        candidates.append(Path(work_root).expanduser() / "jz_datasets")
    home_root = os.getenv("HOME")
    if home_root:
        candidates.append(Path(home_root).expanduser() / ".cache" / "jz_datasets")
    candidates.append(Path("/tmp") / "jz_datasets")

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        return candidate

    raise RuntimeError("Unable to create a real-dataset cache directory.")


def _resolve_dataset(
    target_data: str,
    all_datasets: dict[str, DatasetEntry],
    *,
    dataset_type: str | None = None,
) -> DatasetEntry:
    key = target_data.lower()
    if key not in all_datasets:
        raise KeyError(f"Unknown dataset {target_data!r}. Available datasets: {sorted(all_datasets)}")
    entry = all_datasets[key]
    if dataset_type is not None and entry.dataset_type != dataset_type:
        raise ValueError(f"Dataset {target_data!r} is {entry.dataset_type!r}, expected {dataset_type!r}.")
    return entry


def _standardize_split_arrays(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (x_train - mean) / std, (x_val - mean) / std, (x_test - mean) / std


def _coerce_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_frame = frame.copy().dropna(axis=0).reset_index(drop=True)
    for column in numeric_frame.columns:
        if pd.api.types.is_datetime64_any_dtype(numeric_frame[column]):
            numeric_frame[column] = numeric_frame[column].astype("int64") // 10**9
    non_numeric_columns = [
        column for column in numeric_frame.columns if not pd.api.types.is_numeric_dtype(numeric_frame[column])
    ]
    if non_numeric_columns:
        numeric_frame = pd.get_dummies(numeric_frame, columns=non_numeric_columns, drop_first=False)
    numeric_frame = numeric_frame.replace([np.inf, -np.inf], np.nan).dropna(axis=0).reset_index(drop=True)
    if numeric_frame.shape[1] == 0:
        raise ValueError("No numeric columns remain after preprocessing.")
    return numeric_frame.astype(float)


def split_sample_indices(
    n_rows: int,
    *,
    val_size: float,
    test_size: float,
    random_state: int,
    split_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0 <= val_size < 1:
        raise ValueError("val_size must lie in [0, 1).")
    if not 0 <= test_size < 1:
        raise ValueError("test_size must lie in [0, 1).")
    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size must be < 1.")
    if n_rows < 3:
        raise ValueError("The dataset must contain at least 3 rows.")
    if split_mode not in {"random", "chronological"}:
        raise ValueError("split_mode must be 'random' or 'chronological'.")

    if split_mode == "random":
        indices = np.arange(n_rows)
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )
        val_ratio = val_size / (1.0 - test_size)
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=val_ratio,
            random_state=random_state,
            shuffle=True,
        )
        return train_idx, val_idx, test_idx

    n_test = int(np.floor(test_size * n_rows))
    n_val = int(np.floor(val_size * n_rows))
    n_train = n_rows - n_val - n_test
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError("The requested val/test proportions leave an empty split.")
    return np.arange(0, n_train), np.arange(n_train, n_train + n_val), np.arange(n_train + n_val, n_rows)


def _split_frame_to_tensors(
    frame: pd.DataFrame,
    *,
    split_mode: str,
    val_size: float,
    test_size: float,
    random_state: int,
    standardize: bool,
    device: str | torch.device,
    dtype: torch.dtype,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[Any, Any, Any]]:
    train_idx, val_idx, test_idx = split_sample_indices(
        len(frame),
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        split_mode=split_mode,
    )
    x_train = frame.iloc[train_idx].reset_index(drop=True).to_numpy()
    x_val = frame.iloc[val_idx].reset_index(drop=True).to_numpy()
    x_test = frame.iloc[test_idx].reset_index(drop=True).to_numpy()
    if standardize:
        x_train, x_val, x_test = _standardize_split_arrays(x_train, x_val, x_test)
    return (
        (
            torch.as_tensor(np.array(x_train, copy=True), device=device, dtype=dtype),
            torch.as_tensor(np.array(x_val, copy=True), device=device, dtype=dtype),
            torch.as_tensor(np.array(x_test, copy=True), device=device, dtype=dtype),
        ),
        (train_idx, val_idx, test_idx),
    )


def _split_tensor_to_tensors(
    data: torch.Tensor,
    *,
    split_mode: str,
    val_size: float,
    test_size: float,
    random_state: int,
    standardize: bool,
    device: str | torch.device,
    dtype: torch.dtype,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[Any, Any, Any]]:
    train_idx, val_idx, test_idx = split_sample_indices(
        int(data.shape[0]),
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        split_mode=split_mode,
    )

    def select(indices: np.ndarray) -> torch.Tensor:
        return data.index_select(0, torch.as_tensor(indices, dtype=torch.long))

    x_train = select(train_idx).to(dtype=dtype)
    x_val = select(val_idx).to(dtype=dtype)
    x_test = select(test_idx).to(dtype=dtype)
    if standardize:
        mean = x_train.mean(dim=0, keepdim=True)
        std = x_train.std(dim=0, keepdim=True)
        std = torch.where(std == 0, torch.ones_like(std), std)
        x_train = (x_train - mean) / std
        x_val = (x_val - mean) / std
        x_test = (x_test - mean) / std
    tensors = (
        x_train.to(device=device, dtype=dtype),
        x_val.to(device=device, dtype=dtype),
        x_test.to(device=device, dtype=dtype),
    )
    return tensors, (train_idx, val_idx, test_idx)


def _split_records(records: list[dict[str, Any]], indices: Any) -> list[dict[str, Any]]:
    return [records[int(index)] for index in indices]


def _record_histograms(records: list[dict[str, Any]]) -> dict[str, dict[Any, int]]:
    category_histogram: dict[int, int] = {}
    frequency_histogram: dict[str, int] = {}
    class_histogram: dict[int, int] = {}
    for record in records:
        for category_id in set(int(value) for value in record.get("category_ids", [])):
            category_histogram[category_id] = category_histogram.get(category_id, 0) + 1
        for frequency in set(str(value) for value in record.get("category_frequencies", []) if value):
            frequency_histogram[frequency] = frequency_histogram.get(frequency, 0) + 1
        if "class_id" in record and record["class_id"] is not None:
            class_id = int(record["class_id"])
            class_histogram[class_id] = class_histogram.get(class_id, 0) + 1
    return {
        "category_histogram": category_histogram,
        "frequency_histogram": frequency_histogram,
        "class_histogram": class_histogram,
    }


def _split_metadata(indices: Any, records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    split_meta: dict[str, Any] = {"indices": [int(index) for index in indices], "n_samples": int(len(indices))}
    if records is not None:
        split_records = _split_records(records, indices)
        split_meta["records"] = split_records
        split_meta.update(_record_histograms(split_records))
    return split_meta


def _build_return_metadata(
    *,
    entry: DatasetEntry,
    kind: str,
    target_data: str,
    params: dict[str, Any],
    split_config: dict[str, Any],
    standardize: bool,
    device: str | torch.device,
    dtype: torch.dtype,
    split_indices: tuple[Any, Any, Any],
    payload_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload_metadata = dict(payload_metadata or {})
    records = payload_metadata.pop("records", None)
    train_idx, val_idx, test_idx = split_indices
    return {
        "dataset": entry.metadata(),
        "request": {
            "kind": kind,
            "name": target_data,
            "params": _metadata_value(params),
            "split": _metadata_value(split_config),
            "standardize": bool(standardize),
            "device": str(device),
            "dtype": str(dtype),
        },
        "loader": _metadata_value(payload_metadata),
        "splits": {
            "train": _split_metadata(train_idx, records),
            "val": _split_metadata(val_idx, records),
            "test": _split_metadata(test_idx, records),
        },
    }


def _bool_kwarg(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.lower().strip()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise TypeError(f"{name} must be a boolean.")


def _default_image_cache_dir(dataset_name: str, data_home: Any, cache_dir: Any) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    if data_home is not None:
        return Path(data_home).expanduser() / ".jeanzay_cache" / dataset_name
    cache_root = os.getenv("JEANZAY_CACHE")
    if cache_root:
        return Path(cache_root).expanduser() / dataset_name
    if os.getenv("WORK"):
        return Path(os.environ["WORK"]).expanduser() / "jeanzay_cache" / dataset_name
    if os.getenv("XDG_CACHE_HOME"):
        return Path(os.environ["XDG_CACHE_HOME"]).expanduser() / "jeanzaydata" / dataset_name
    return Path.home() / ".cache" / "jeanzaydata" / dataset_name


def _processed_image_cache_path(
    dataset_name: str,
    source_path: Path,
    *,
    options: dict[str, Any],
    cache_dir: Any,
    data_home: Any,
) -> Path:
    stat = source_path.stat()
    key = {
        "version": _IMAGE_CACHE_VERSION,
        "dataset_name": dataset_name,
        "source_path": str(source_path.resolve()),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        **options,
    }
    digest = hashlib.sha256(repr(sorted(key.items())).encode("utf-8")).hexdigest()[:24]
    return _default_image_cache_dir(dataset_name, data_home, cache_dir) / f"{dataset_name}_{digest}.pt"


def _load_processed_image_cache(cache_path: Path) -> DatasetPayload | None:
    try:
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if not isinstance(cached, dict) or not isinstance(cached.get("data"), torch.Tensor):
            return None
        metadata = dict(cached.get("metadata", {}))
        metadata["cache_hit"] = True
        metadata["cache_path"] = str(cache_path)
        return DatasetPayload(data=cached["data"].to(dtype=torch.float32), metadata=metadata)
    except Exception as exc:
        warnings.warn(f"Ignoring unreadable image cache at {cache_path}: {exc}", RuntimeWarning, stacklevel=2)
        return None


def _write_processed_image_cache(cache_path: Path, payload: DatasetPayload) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        torch.save({"data": payload.data.cpu(), "metadata": payload.metadata}, tmp_path)
        os.replace(tmp_path, cache_path)
    except Exception as exc:
        warnings.warn(f"Could not write image cache at {cache_path}: {exc}", RuntimeWarning, stacklevel=2)


def _print_image_loader_progress(dataset_name: str, current: int, total: int, started_at: float) -> None:
    elapsed = time.perf_counter() - started_at
    print(
        f"[dataset] image {dataset_name:12s} {current}/{total} elapsed={elapsed:.1f}s",
        flush=True,
    )


def _load_lvis(**kwargs: Any) -> DatasetPayload:
    split = _normalize_lvis_split(str(kwargs.pop("split", "train")))
    image_size = int(kwargs.pop("image_size", 64))
    max_samples_raw = kwargs.pop("max_samples", None)
    max_samples = None if max_samples_raw is None else int(max_samples_raw)
    category_frequency_raw = kwargs.pop("category_frequency", None)
    category_frequency = None if category_frequency_raw is None else str(category_frequency_raw)
    seed = int(kwargs.pop("seed", 0))
    data_home = kwargs.pop("data_home", None)
    cache = _bool_kwarg(kwargs.pop("cache", True), name="cache")
    cache_dir = kwargs.pop("cache_dir", None)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected LVIS loader kwargs: {unexpected}.")
    if image_size <= 0:
        raise ValueError("image_size must be > 0.")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be > 0 when provided.")

    base_roots = _lvis_base_roots(data_home)
    annotation_path, _ = _find_lvis_annotation(base_roots, split)
    cache_path = _processed_image_cache_path(
        "lvis",
        annotation_path,
        options={
            "split": split,
            "image_size": image_size,
            "max_samples": max_samples,
            "category_frequency": category_frequency,
            "seed": seed,
        },
        cache_dir=cache_dir,
        data_home=data_home,
    ) if cache else None
    if cache_path is not None and cache_path.is_file():
        cached_payload = _load_processed_image_cache(cache_path)
        if cached_payload is not None:
            return cached_payload

    data = _load_lvis_json(annotation_path)
    images = _lvis_candidate_images(data, category_frequency=category_frequency, annotation_path=annotation_path)
    images = _select_lvis_images(images, max_samples=max_samples, seed=seed)
    if not images:
        raise RuntimeError(
            f"No LVIS images matched split={split!r} category_frequency={category_frequency!r} in {annotation_path}."
        )

    search_roots = _lvis_search_roots(base_roots, annotation_path=annotation_path)
    image_dirs = _find_lvis_image_dirs(search_roots)
    name_by_id, frequency_by_id = _lvis_category_maps(data)
    category_ids_by_image = _lvis_category_ids_by_image(data)
    tensors = []
    records = []
    started_at = time.perf_counter()
    total_images = len(images)
    print(
        f"[dataset] image lvis         start total={total_images} split={split} image_size={image_size}",
        flush=True,
    )
    for index, image in enumerate(images, start=1):
        image_path = _resolve_lvis_image_path(
            image,
            split=split,
            image_dirs=image_dirs,
            search_roots=search_roots,
        )
        tensors.append(read_rgb_resized(image_path, image_size))
        image_id = int(image["id"])
        records.append(
            _lvis_record(
                image,
                image_path,
                category_ids_by_image.get(image_id, []),
                name_by_id=name_by_id,
                frequency_by_id=frequency_by_id,
            )
        )
        if index == total_images or index == 1 or index % 1000 == 0:
            _print_image_loader_progress("lvis", index, total_images, started_at)
    metadata = {
        "source_split": split,
        "annotation_path": str(annotation_path),
        "image_size": int(image_size),
        "category_frequency_filter": category_frequency,
        "n_selected_images": len(records),
        "records": records,
        "cache_hit": False,
    }
    if cache_path is not None:
        metadata["cache_path"] = str(cache_path)
    payload = DatasetPayload(data=torch.stack(tensors, dim=0).to(dtype=torch.float32), metadata=metadata)
    if cache_path is not None:
        _write_processed_image_cache(cache_path, payload)
    return payload


def _load_cifar100_lt(**kwargs: Any) -> DatasetPayload:
    split = _normalize_cifar100_split(str(kwargs.pop("split", "train")))
    image_size = int(kwargs.pop("image_size", 64))
    imbalance_factor = float(kwargs.pop("imbalance_factor", 100))
    max_samples_raw = kwargs.pop("max_samples", None)
    max_samples = None if max_samples_raw is None else int(max_samples_raw)
    seed = int(kwargs.pop("seed", 0))
    data_home = kwargs.pop("data_home", None)
    cache = _bool_kwarg(kwargs.pop("cache", True), name="cache")
    cache_dir = kwargs.pop("cache_dir", None)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected CIFAR-100-LT loader kwargs: {unexpected}.")
    if image_size <= 0:
        raise ValueError("image_size must be > 0.")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be > 0 when provided.")

    root = _cifar100_root(data_home)
    source_path = (root / "cifar-100-python") if (root / "cifar-100-python").is_dir() else root
    cache_path = _processed_image_cache_path(
        "cifar100_lt",
        source_path,
        options={
            "split": split,
            "image_size": image_size,
            "imbalance_factor": float(imbalance_factor),
            "max_samples": max_samples,
            "seed": seed,
        },
        cache_dir=cache_dir,
        data_home=data_home,
    ) if cache else None
    if cache_path is not None and cache_path.is_file():
        cached_payload = _load_processed_image_cache(cache_path)
        if cached_payload is not None:
            return cached_payload

    images, labels, indices, label_names, class_counts = load_cifar100_lt_arrays(
        root=root,
        split=split,
        imbalance_factor=imbalance_factor,
        max_samples=max_samples,
        seed=seed,
    )
    if images.shape[0] == 0:
        raise RuntimeError(
            f"CIFAR-100-LT produced no samples for split={split!r} imbalance_factor={imbalance_factor}."
        )

    tensors = []
    records = []
    started_at = time.perf_counter()
    total_images = int(images.shape[0])
    print(
        f"[dataset] image cifar100_lt  start total={total_images} split={split} image_size={image_size} "
        f"imbalance_factor={imbalance_factor}",
        flush=True,
    )
    histogram: dict[int, int] = {}
    for index in range(total_images):
        tensors.append(array_uint8_to_resized_tensor(images[index], image_size))
        class_id = int(labels[index])
        histogram[class_id] = histogram.get(class_id, 0) + 1
        records.append(
            {
                "source_index": int(indices[index]),
                "class_id": class_id,
                "class_name": label_names[class_id] if class_id < len(label_names) else str(class_id),
            }
        )
        if index + 1 == total_images or index == 0 or (index + 1) % 1000 == 0:
            _print_image_loader_progress("cifar100_lt", index + 1, total_images, started_at)

    metadata = {
        "source_split": split,
        "source_path": str(source_path),
        "image_size": int(image_size),
        "imbalance_factor": float(imbalance_factor),
        "n_selected_images": total_images,
        "labels": [int(value) for value in labels],
        "selected_indices": [int(value) for value in indices],
        "long_tail_class_counts": [int(count) for count in class_counts],
        "class_histogram": histogram,
        "label_names": list(label_names),
        "records": records,
        "cache_hit": False,
    }
    if cache_path is not None:
        metadata["cache_path"] = str(cache_path)
    payload = DatasetPayload(data=torch.stack(tensors, dim=0).to(dtype=torch.float32), metadata=metadata)
    if cache_path is not None:
        _write_processed_image_cache(cache_path, payload)
    return payload


def _load_imagenet_lt(**kwargs: Any) -> DatasetPayload:
    split = _normalize_imagenet_lt_split(str(kwargs.pop("split", "train")))
    image_size = int(kwargs.pop("image_size", 64))
    max_samples_raw = kwargs.pop("max_samples", None)
    max_samples = None if max_samples_raw is None else int(max_samples_raw)
    seed = int(kwargs.pop("seed", 0))
    data_home = kwargs.pop("data_home", None)
    annotation_path_arg = kwargs.pop("annotation_path", None)
    imagenet_root_arg = kwargs.pop("imagenet_root", None)
    cache = _bool_kwarg(kwargs.pop("cache", True), name="cache")
    cache_dir = kwargs.pop("cache_dir", None)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected ImageNet-LT loader kwargs: {unexpected}.")
    if image_size <= 0:
        raise ValueError("image_size must be > 0.")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be > 0 when provided.")

    annotation_path = _resolve_imagenet_lt_annotation(
        data_home=data_home,
        split=split,
        annotation_path=annotation_path_arg,
    )
    imagenet_root = _resolve_imagenet_root(data_home=data_home, imagenet_root=imagenet_root_arg)
    cache_path = _processed_image_cache_path(
        "imagenet_lt",
        annotation_path,
        options={
            "split": split,
            "image_size": image_size,
            "max_samples": max_samples,
            "seed": seed,
            "imagenet_root": str(imagenet_root.resolve()),
        },
        cache_dir=cache_dir,
        data_home=data_home,
    ) if cache else None
    if cache_path is not None and cache_path.is_file():
        cached_payload = _load_processed_image_cache(cache_path)
        if cached_payload is not None:
            return cached_payload

    raw_records = _parse_imagenet_lt_split_file(annotation_path)
    raw_records = _select_imagenet_lt_records(raw_records, max_samples=max_samples, seed=seed)
    if not raw_records:
        raise RuntimeError(
            f"ImageNet-LT produced no records for split={split!r} in {annotation_path}."
        )

    total_records = len(raw_records)
    data = torch.empty((total_records, 3, image_size, image_size), dtype=torch.float32)
    records = []
    histogram: dict[int, int] = {}
    started_at = time.perf_counter()
    print(
        f"[dataset] image imagenet_lt  start total={total_records} split={split} "
        f"image_size={image_size} root={imagenet_root}",
        flush=True,
    )
    for index, raw_record in enumerate(raw_records, start=1):
        image_path = _resolve_imagenet_lt_image_path(raw_record, imagenet_root)
        data[index - 1].copy_(read_rgb_resized(image_path, image_size))
        record = _imagenet_lt_record(raw_record, image_path)
        records.append(record)
        class_id = int(record["class_id"])
        histogram[class_id] = histogram.get(class_id, 0) + 1
        if index == total_records or index == 1 or index % 1000 == 0:
            _print_image_loader_progress("imagenet_lt", index, total_records, started_at)

    metadata = {
        "source_split": split,
        "annotation_path": str(annotation_path),
        "imagenet_root": str(imagenet_root),
        "image_size": int(image_size),
        "n_selected_images": total_records,
        "labels": [int(record["class_id"]) for record in records],
        "class_histogram": histogram,
        "records": records,
        "cache_hit": False,
    }
    if cache_path is not None:
        metadata["cache_path"] = str(cache_path)
    payload = DatasetPayload(data=data, metadata=metadata)
    if cache_path is not None:
        _write_processed_image_cache(cache_path, payload)
    return payload


def _load_hrrr(**kwargs: Any) -> torch.Tensor:
    filename = "hrrr_apcp_100x100.pt"
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected HRRR loader kwargs: {unexpected}.")
    store_root = os.getenv("STORE")
    if not store_root:
        raise RuntimeError(
            "$STORE is not set. "
            f"Expected the precomputed HRRR tensor at $STORE/hrrr_data/{filename}."
        )
    path = Path(store_root).expanduser() / "hrrr_data" / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"HRRR tensor file not found at {path}. "
            f"Expected the precomputed HRRR tensor at $STORE/hrrr_data/{filename}."
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        if "frames" in payload:
            payload = payload["frames"]
        elif "apcp_mm" in payload:
            payload = payload["apcp_mm"]
        else:
            raise ValueError(
                f"Unexpected HRRR payload keys in {path}: {sorted(payload)}. "
                "Expected a tensor payload or a dict containing 'frames' or 'apcp_mm'."
            )
    tensor = torch.as_tensor(payload, dtype=torch.float32)
    if tensor.ndim != 4 or tuple(tensor.shape[1:]) != (1, 100, 100):
        raise ValueError(
            f"Unexpected HRRR tensor shape in {path}: {tuple(tensor.shape)}. Expected (N, 1, 100, 100)."
        )
    valid_samples = torch.isfinite(tensor).flatten(start_dim=1).all(dim=1)
    return tensor[valid_samples].contiguous()


REAL_DATASETS: dict[str, DatasetEntry] = {
    "lvis": DatasetEntry(
        name="lvis",
        dataset_type="real",
        description="LVIS long-tailed object categories from local Jean Zay COCO/LVIS files; default image_size=64.",
        standardize_default=False,
        dim=(3, 64, 64),
        loader=_load_lvis,
    ),
    "cifar100_lt": DatasetEntry(
        name="cifar100_lt",
        dataset_type="real",
        description="CIFAR-100 reshaped into a long-tailed subset using exponential class decay; default image_size=64.",
        standardize_default=False,
        dim=(3, 64, 64),
        loader=_load_cifar100_lt,
    ),
    "imagenet_lt": DatasetEntry(
        name="imagenet_lt",
        dataset_type="real",
        description="ImageNet-LT split using shipped annotation files and a local ImageNet image tree; default image_size=64.",
        standardize_default=False,
        dim=(3, 64, 64),
        loader=_load_imagenet_lt,
    ),
    "hrrr": DatasetEntry(
        name="hrrr",
        dataset_type="real",
        description="HRRR accumulated precipitation fields on a 100x100 crop.",
        standardize_default=False,
        dim=(1, 100, 100),
        loader=_load_hrrr,
    ),
}


def fetch_real_data(target_data: str, **kwargs: Any):
    return_metadata = bool(kwargs.pop("return_metadata", False))
    entry = _resolve_dataset(target_data, REAL_DATASETS, dataset_type="real")
    if entry.loader is None:
        raise RuntimeError(f"Real dataset {target_data!r} has no loader.")

    n_samples = kwargs.pop("n_samples", None)
    if n_samples is not None:
        n_samples = int(n_samples)
    entry_name = getattr(entry, "name", target_data)
    if entry_name in {"lvis", "cifar100_lt", "imagenet_lt"} and n_samples is not None and "max_samples" not in kwargs:
        kwargs["max_samples"] = n_samples
        n_samples = None

    val_size = float(kwargs.pop("val_size", 0.15))
    test_size = float(kwargs.pop("test_size", 0.15))
    random_state = int(kwargs.pop("random_state", 0))
    standardize_value = kwargs.pop("standardize", None)
    standardize = bool(entry.standardize_default if standardize_value is None else standardize_value)
    device = kwargs.pop("device", "cpu")
    dtype = kwargs.pop("dtype", torch.float32)

    loaded = entry.loader(**kwargs)
    payload_metadata: dict[str, Any] = {}
    if isinstance(loaded, DatasetPayload):
        payload_metadata = dict(loaded.metadata)
        loaded = loaded.data

    if isinstance(loaded, pd.DataFrame):
        frame = _coerce_numeric_frame(loaded)
        records = payload_metadata.get("records")
        if n_samples is not None:
            if n_samples > len(frame):
                warnings.warn(
                    f"n_samples={n_samples} exceeds available samples ({len(frame)}); returning all.",
                    stacklevel=2,
                )
            selected = frame.sample(n=min(n_samples, len(frame)), random_state=random_state).index.to_numpy()
            frame = frame.loc[selected].reset_index(drop=True)
            if records is not None:
                payload_metadata["records"] = _split_records(records, selected)
        tensors, split_indices = _split_frame_to_tensors(
            frame,
            split_mode=entry.split_mode,
            val_size=val_size,
            test_size=test_size,
            random_state=random_state,
            standardize=standardize,
            device=device,
            dtype=dtype,
        )
    else:
        data = torch.as_tensor(loaded)
        records = payload_metadata.get("records")
        if n_samples is not None:
            if n_samples > data.shape[0]:
                warnings.warn(
                    f"n_samples={n_samples} exceeds available samples ({data.shape[0]}); returning all.",
                    stacklevel=2,
                )
            rng = torch.Generator().manual_seed(random_state)
            idx = torch.randperm(data.shape[0], generator=rng)[: min(n_samples, data.shape[0])]
            data = data[idx]
            if records is not None:
                payload_metadata["records"] = _split_records(records, idx.tolist())
        tensors, split_indices = _split_tensor_to_tensors(
            data,
            val_size=val_size,
            test_size=test_size,
            random_state=random_state,
            split_mode=entry.split_mode,
            standardize=standardize,
            device=device,
            dtype=dtype,
        )

    if not return_metadata:
        return tensors
    metadata = _build_return_metadata(
        entry=entry,
        kind="real",
        target_data=target_data,
        params=kwargs,
        split_config={"val_size": val_size, "test_size": test_size, "random_state": random_state},
        standardize=standardize,
        device=device,
        dtype=dtype,
        split_indices=split_indices,
        payload_metadata=payload_metadata,
    )
    return (*tensors, metadata)


def get_dataset_metadata(target_data: str) -> dict[str, Any]:
    return _resolve_dataset(target_data, REAL_DATASETS).metadata()


def list_datasets() -> list[str]:
    return sorted(REAL_DATASETS)
