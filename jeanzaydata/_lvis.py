"""LVIS-specific dataset helpers."""

import json
import os
from pathlib import Path
from typing import Any
import numpy as np

_LVIS_ANNOTATION_FILES: dict[str, tuple[str, ...]] = {
    "train": ("lvis_v1_train.json", "lvis_v0.5_train.json"),
    "val": ("lvis_v1_val.json", "lvis_v0.5_val.json"),
}
_LVIS_IMAGE_DIR_NAMES = ("train2017", "val2017")


def _normalize_lvis_split(split: str) -> str:
    split_key = str(split).lower().strip()
    if split_key in {"validation", "valid"}:
        split_key = "val"
    if split_key not in _LVIS_ANNOTATION_FILES:
        raise ValueError(f"Unsupported LVIS split {split!r}. Expected one of {sorted(_LVIS_ANNOTATION_FILES)}.")
    return split_key


def _lvis_base_roots(data_home: str | Path | None) -> list[Path]:
    if data_home is not None:
        roots = [Path(data_home).expanduser()]
    else:
        dsdir = os.getenv("DSDIR")
        if not dsdir:
            raise RuntimeError(
                "LVIS data_home was not provided and $DSDIR is not set. "
                "This loader is Jean-Zay/local-files only and will not download LVIS assets."
            )
        roots = [Path(dsdir).expanduser()]

    missing = [str(root) for root in roots if not root.exists()]
    if missing:
        raise RuntimeError(f"LVIS search root(s) do not exist: {missing}")
    return roots


def _unique_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        expanded = path.expanduser()
        try:
            key = expanded.resolve()
        except OSError:
            key = expanded.absolute()
        if key in seen or not expanded.exists():
            continue
        seen.add(key)
        unique.append(expanded)
    return unique


def _lvis_search_roots(base_roots: list[Path], annotation_path: Path | None = None) -> list[Path]:
    roots: list[Path] = []
    if annotation_path is not None:
        roots.extend([annotation_path.parent, annotation_path.parent.parent])
    for root in base_roots:
        roots.extend(
            [
                root / "annotations",
                root / "lvis",
                root / "LVIS",
                root / "lvis" / "annotations",
                root / "LVIS" / "annotations",
                root / "coco",
                root / "COCO",
                root / "coco" / "annotations",
                root / "COCO" / "annotations",
                root / "images",
                root / "coco" / "images",
                root / "COCO" / "images",
                root,
            ]
        )
    return _unique_paths(roots)


def _find_named_paths(
    roots: list[Path],
    names: set[str],
    *,
    want_dir: bool,
    max_depth: int = 6,
) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root.is_file():
            if not want_dir and root.name in names:
                resolved = root.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    found.append(root)
            continue

        if want_dir and root.name in names:
            resolved = root.resolve()
            if resolved not in seen:
                seen.add(resolved)
                found.append(root)

        for current, dirs, files in os.walk(root):
            current_path = Path(current)
            try:
                depth = len(current_path.relative_to(root).parts)
            except ValueError:
                depth = 0
            if depth >= max_depth:
                dirs[:] = []
            dirs[:] = [name for name in dirs if not name.startswith(".") and name != "__pycache__"]

            candidates = dirs if want_dir else files
            for name in candidates:
                if name not in names:
                    continue
                path = current_path / name
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                found.append(path)
    return found


def _find_lvis_annotation(base_roots: list[Path], split: str) -> tuple[Path, list[Path]]:
    roots = _lvis_search_roots(base_roots)
    names = set(_LVIS_ANNOTATION_FILES[split])
    for root in roots:
        if root.is_file() and root.name in names:
            return root, roots
        if root.is_dir():
            for name in _LVIS_ANNOTATION_FILES[split]:
                candidate = root / name
                if candidate.is_file():
                    return candidate, roots

    broad_roots = set()
    for root in base_roots:
        try:
            broad_roots.add(root.resolve())
        except OSError:
            broad_roots.add(root.absolute())
    narrow_roots = []
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            resolved = root.absolute()
        if resolved not in broad_roots:
            narrow_roots.append(root)
    matches = _find_named_paths(narrow_roots, names, want_dir=False, max_depth=3)
    if matches:
        return matches[0], roots
    searched_roots = "\n".join(f"  - {root}" for root in roots)
    raise RuntimeError(
        "Could not find LVIS annotation file. "
        f"Searched for {', '.join(sorted(_LVIS_ANNOTATION_FILES[split]))} under:\n{searched_roots}"
    )


def _load_lvis_json(annotation_path: Path) -> dict[str, Any]:
    with annotation_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise RuntimeError(f"LVIS annotation file {annotation_path} did not decode to a JSON object.")
    if not isinstance(data.get("images"), list):
        raise RuntimeError(f"LVIS annotation file {annotation_path} has no 'images' list.")
    return data


def _lvis_candidate_images(
    data: dict[str, Any],
    *,
    category_frequency: str | None,
    annotation_path: Path,
) -> list[dict[str, Any]]:
    images_by_id: dict[int, dict[str, Any]] = {}
    ordered_ids: list[int] = []
    for image in data.get("images", []):
        if not isinstance(image, dict) or "id" not in image:
            continue
        image_id = int(image["id"])
        if image_id in images_by_id:
            continue
        images_by_id[image_id] = image
        ordered_ids.append(image_id)

    if category_frequency is None:
        return [images_by_id[image_id] for image_id in ordered_ids]

    frequency = str(category_frequency).lower().strip()
    if frequency not in {"r", "c", "f"}:
        raise ValueError("category_frequency must be one of 'r', 'c', 'f', or None.")

    category_frequency_by_id: dict[int, str] = {}
    for category in data.get("categories", []):
        if not isinstance(category, dict) or "id" not in category:
            continue
        label = category.get("frequency", category.get("freq"))
        if label is not None:
            category_frequency_by_id[int(category["id"])] = str(label).lower()

    if not category_frequency_by_id:
        raise RuntimeError(
            f"category_frequency={frequency!r} was requested, but {annotation_path} "
            "does not expose LVIS category frequency labels."
        )

    matching_ids: set[int] = set()
    for annotation in data.get("annotations", []):
        if not isinstance(annotation, dict):
            continue
        category_id = annotation.get("category_id")
        image_id = annotation.get("image_id")
        if category_id is None or image_id is None:
            continue
        if category_frequency_by_id.get(int(category_id)) == frequency:
            matching_ids.add(int(image_id))

    return [images_by_id[image_id] for image_id in ordered_ids if image_id in matching_ids]


def _lvis_category_maps(data: dict[str, Any]) -> tuple[dict[int, str], dict[int, str]]:
    name_by_id: dict[int, str] = {}
    frequency_by_id: dict[int, str] = {}
    for category in data.get("categories", []):
        if not isinstance(category, dict) or "id" not in category:
            continue
        category_id = int(category["id"])
        name_by_id[category_id] = str(category.get("name", category_id))
        label = category.get("frequency", category.get("freq"))
        if label is not None:
            frequency_by_id[category_id] = str(label).lower()
    return name_by_id, frequency_by_id


def _lvis_category_ids_by_image(data: dict[str, Any]) -> dict[int, list[int]]:
    categories_by_image: dict[int, list[int]] = {}
    seen_pairs: set[tuple[int, int]] = set()
    for annotation in data.get("annotations", []):
        if not isinstance(annotation, dict):
            continue
        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")
        if image_id is None or category_id is None:
            continue
        pair = (int(image_id), int(category_id))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        categories_by_image.setdefault(pair[0], []).append(pair[1])
    return categories_by_image


def _lvis_record(
    image: dict[str, Any],
    image_path: Path,
    category_ids: list[int],
    *,
    name_by_id: dict[int, str],
    frequency_by_id: dict[int, str],
) -> dict[str, Any]:
    file_name = str(image.get("file_name") or image_path.name)
    return {
        "image_id": int(image["id"]),
        "file_name": file_name,
        "path": str(image_path),
        "category_ids": [int(category_id) for category_id in category_ids],
        "category_names": [name_by_id.get(int(category_id), str(category_id)) for category_id in category_ids],
        "category_frequencies": [
            frequency_by_id[int(category_id)]
            for category_id in category_ids
            if int(category_id) in frequency_by_id
        ],
    }


def _select_lvis_images(
    images: list[dict[str, Any]],
    *,
    max_samples: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if max_samples is None or len(images) <= max_samples:
        return images
    rng = np.random.default_rng(int(seed))
    selected = np.sort(rng.choice(len(images), size=max_samples, replace=False))
    return [images[int(idx)] for idx in selected]


def _find_lvis_image_dirs(search_roots: list[Path]) -> dict[str, list[Path]]:
    by_name: dict[str, list[Path]] = {name: [] for name in _LVIS_IMAGE_DIR_NAMES}
    seen: set[Path] = set()

    def add(path: Path) -> None:
        if not path.is_dir() or path.name not in by_name:
            return
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path.absolute()
        if resolved in seen:
            return
        seen.add(resolved)
        by_name[path.name].append(path)

    for root in search_roots:
        for name in _LVIS_IMAGE_DIR_NAMES:
            add(root if root.name == name else root / name)
            add(root / "images" / name)
            add(root / "coco" / name)
            add(root / "COCO" / name)
            add(root / "coco" / "images" / name)
            add(root / "COCO" / "images" / name)

    if any(by_name.values()):
        return by_name

    narrow_roots = [
        root
        for root in search_roots
        if root.name in {"lvis", "LVIS", "coco", "COCO", "images"}
    ]
    for path in _find_named_paths(narrow_roots, set(_LVIS_IMAGE_DIR_NAMES), want_dir=True, max_depth=3):
        add(path)
    return by_name


def _lvis_url_parts(value: Any) -> tuple[str | None, str | None]:
    if not value:
        return None, None
    parts = str(value).rstrip("/").split("/")
    if len(parts) < 2:
        return None, Path(parts[-1]).name if parts else None
    return parts[-2], Path(parts[-1]).name


def _resolve_lvis_image_path(
    image: dict[str, Any],
    *,
    split: str,
    image_dirs: dict[str, list[Path]],
    search_roots: list[Path],
) -> Path:
    file_name_raw = image.get("file_name")
    file_name = str(file_name_raw) if file_name_raw else ""
    file_path = Path(file_name) if file_name else None
    basename = file_path.name if file_path is not None and file_path.name else ""
    url_folder, url_name = _lvis_url_parts(image.get("coco_url"))
    if not basename:
        basename = url_name or f"{int(image['id']):012d}.jpg"

    candidates: list[Path] = []
    if file_path is not None:
        if file_path.is_absolute():
            candidates.append(file_path)
        else:
            candidates.extend(root / file_path for root in search_roots)

    seen_folders: set[str] = set()
    for folder_name in [url_folder, f"{split}2017", *_LVIS_IMAGE_DIR_NAMES]:
        if not folder_name or folder_name in seen_folders:
            continue
        seen_folders.add(folder_name)
        for image_dir in image_dirs.get(folder_name, []):
            candidates.append(image_dir / basename)

    for dirs in image_dirs.values():
        candidates.extend(image_dir / basename for image_dir in dirs)

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate.absolute()
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.exists() and candidate.is_file():
            return candidate

    searched = "\n".join(f"  - {root}" for root in search_roots)
    raise RuntimeError(
        f"Could not resolve LVIS image for image id={image.get('id')} file_name={file_name!r}. "
        f"Searched image roots:\n{searched}"
    )
