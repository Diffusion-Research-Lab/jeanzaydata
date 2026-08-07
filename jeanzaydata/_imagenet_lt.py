"""ImageNet-LT loader using shipped annotation files and a local ImageNet tree."""

import os
from pathlib import Path
from typing import Any
import numpy as np

_IMAGENET_LT_SPLIT_FILES = {
    "train": "ImageNet_LT_train.txt",
    "val": "ImageNet_LT_val.txt",
    "test": "ImageNet_LT_test.txt",
}
_DEFAULT_IMAGENET_ROOT = "/lustre/fswork/dataset/imagenet"


def _normalize_imagenet_lt_split(split: str) -> str:
    key = str(split).lower().strip()
    if key in {"validation", "valid"}:
        key = "val"
    if key not in _IMAGENET_LT_SPLIT_FILES:
        raise ValueError(
            f"Unsupported ImageNet-LT split {split!r}. Expected one of {sorted(_IMAGENET_LT_SPLIT_FILES)}."
        )
    return key


def _imagenet_lt_root(data_home: str | Path | None) -> Path:
    if data_home is not None:
        root = Path(data_home).expanduser()
    else:
        data_root = os.getenv("JEANZAY_DATA")
        if not data_root:
            raise RuntimeError(
                "ImageNet-LT raw data location is not configured. "
                "Set JEANZAY_DATA so the loader uses $JEANZAY_DATA/raw/imagenet_lt, "
                "or pass data_home explicitly. "
                "Run `python -m jeanzaydata init imagenet_lt` after exporting JEANZAY_DATA."
            )
        root = Path(data_root).expanduser() / "raw" / "imagenet_lt"
    if not root.is_dir():
        raise RuntimeError(
            f"ImageNet-LT root does not exist: {root}. "
            "Run `python -m jeanzaydata init imagenet_lt` to populate it."
        )
    return root


def _resolve_imagenet_lt_annotation(
    *,
    data_home: str | Path | None,
    split: str,
    annotation_path: str | Path | None,
) -> Path:
    if annotation_path is not None:
        candidate = Path(annotation_path).expanduser()
        if not candidate.is_file():
            raise RuntimeError(f"ImageNet-LT annotation file not found: {candidate}")
        return candidate
    root = _imagenet_lt_root(data_home)
    candidate = root / "annotations" / _IMAGENET_LT_SPLIT_FILES[split]
    if not candidate.is_file():
        raise RuntimeError(
            f"ImageNet-LT annotation file not found: {candidate}. "
            "Run `python -m jeanzaydata init imagenet_lt` to copy the bundled annotations."
        )
    return candidate


def _resolve_imagenet_root(
    *,
    data_home: str | Path | None,
    imagenet_root: str | Path | None,
) -> Path:
    candidates: list[Path] = []
    if imagenet_root is not None:
        candidates.append(Path(imagenet_root).expanduser())
    if data_home is not None:
        candidates.append(Path(data_home).expanduser() / "imagenet")
    data_root = os.getenv("JEANZAY_DATA")
    if data_root:
        candidates.append(Path(data_root).expanduser() / "raw" / "imagenet_lt" / "imagenet")
    candidates.append(Path(_DEFAULT_IMAGENET_ROOT))

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate.absolute()
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.is_dir():
            return candidate
    searched = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise RuntimeError(f"Could not locate an ImageNet image root. Searched:\n{searched}")


def _parse_imagenet_lt_split_file(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise RuntimeError(
                    f"Malformed ImageNet-LT line {line_no} in {path}: {raw_line!r}"
                )
            relative_path = " ".join(parts[:-1])
            try:
                class_id = int(parts[-1])
            except ValueError as exc:
                raise RuntimeError(
                    f"Malformed ImageNet-LT class id at line {line_no} in {path}: {raw_line!r}"
                ) from exc
            rel = Path(relative_path)
            wnid = rel.parent.name if rel.parent.name else ""
            records.append(
                {
                    "relative_path": relative_path,
                    "wnid": wnid,
                    "class_id": class_id,
                    "file_name": rel.name,
                }
            )
    if not records:
        raise RuntimeError(f"ImageNet-LT annotation file is empty: {path}")
    return records


def _select_imagenet_lt_records(
    records: list[dict[str, Any]],
    *,
    max_samples: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if max_samples is None or len(records) <= max_samples:
        return records
    rng = np.random.default_rng(int(seed))
    selected = np.sort(rng.choice(len(records), size=int(max_samples), replace=False))
    return [records[int(idx)] for idx in selected]


def _resolve_imagenet_lt_image_path(record: dict[str, Any], imagenet_root: Path) -> Path:
    relative_path = Path(str(record["relative_path"]))
    if relative_path.is_absolute():
        if relative_path.is_file():
            return relative_path
        raise RuntimeError(f"ImageNet-LT image not found at absolute path: {relative_path}")
    candidate = imagenet_root / relative_path
    if candidate.is_file():
        return candidate
    raise RuntimeError(
        f"ImageNet-LT image not found: {candidate}. Annotation entry: {record['relative_path']!r}."
    )


def _imagenet_lt_record(record: dict[str, Any], image_path: Path) -> dict[str, Any]:
    return {
        "relative_path": str(record["relative_path"]),
        "wnid": str(record["wnid"]),
        "file_name": str(record["file_name"]),
        "class_id": int(record["class_id"]),
        "path": str(image_path),
    }
