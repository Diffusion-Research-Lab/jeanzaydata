"""Tests for Jean Zay real-dataset loaders."""

import json
import numpy as np
import os
import pickle
from types import SimpleNamespace
import pandas as pd
import pytest
import torch
from jeanzaydata._dataset import DatasetPayload, fetch_real_data, get_dataset_metadata as get_real_dataset_metadata, list_datasets as list_real_datasets, _load_cifar100_lt, _load_hrrr, _load_imagenet_lt, _load_lvis, _resolve_real_data_home, _standardize_split_arrays, _resolve_dataset, split_sample_indices


def _write_lvis_fixture(root):
    image_mod = pytest.importorskip("PIL.Image")
    annotations_dir = root / "annotations"
    image_dir = root / "coco" / "train2017"
    annotations_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)

    images = []
    for image_id in range(1, 7):
        file_name = f"{image_id:012d}.jpg"
        color = (image_id * 30 % 255, image_id * 40 % 255, image_id * 50 % 255)
        image_mod.new("RGB", (10, 12), color=color).save(image_dir / file_name)
        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "coco_url": f"http://images.cocodataset.org/train2017/{file_name}",
            }
        )

    payload = {
        "images": images,
        "categories": [
            {"id": 1, "name": "rare thing", "frequency": "r"},
            {"id": 2, "name": "frequent thing", "frequency": "f"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1},
            {"id": 2, "image_id": 1, "category_id": 1},
            {"id": 3, "image_id": 2, "category_id": 2},
            {"id": 4, "image_id": 3, "category_id": 1},
            {"id": 5, "image_id": 4, "category_id": 2},
            {"id": 6, "image_id": 5, "category_id": 1},
            {"id": 7, "image_id": 6, "category_id": 2},
        ],
    }
    (annotations_dir / "lvis_v1_train.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


def test_resolve_real_data_home_prefers_work(tmp_path, monkeypatch):
    work_root = tmp_path / "work"
    home_root = tmp_path / "home"
    monkeypatch.delenv("JEANZAY_DATA", raising=False)
    monkeypatch.delenv("JEANZAY_DATA_HOME", raising=False)
    monkeypatch.setenv("WORK", str(work_root))
    monkeypatch.setenv("HOME", str(home_root))

    resolved = _resolve_real_data_home()

    assert resolved == work_root / "jz_datasets"


def test_resolve_real_data_home_uses_home_when_work_is_missing(tmp_path, monkeypatch):
    home_root = tmp_path / "home"
    monkeypatch.delenv("JEANZAY_DATA", raising=False)
    monkeypatch.delenv("JEANZAY_DATA_HOME", raising=False)
    monkeypatch.delenv("WORK", raising=False)
    monkeypatch.setenv("HOME", str(home_root))

    resolved = _resolve_real_data_home()

    assert resolved == home_root / ".cache" / "jz_datasets"


def test_load_hrrr_reads_fixed_store_file_and_drops_nonfinite_samples(tmp_path, monkeypatch):
    hrrr_dir = tmp_path / "hrrr_data"
    hrrr_dir.mkdir()
    values = torch.arange(2 * 1 * 100 * 100, dtype=torch.float64).reshape(2, 1, 100, 100)
    values[0, 0, 0, 0] = float("nan")
    values[0, 0, 0, 1] = float("inf")
    values[0, 0, 0, 2] = float("-inf")
    torch.save(values, hrrr_dir / "hrrr_apcp_100x100.pt")
    monkeypatch.setenv("STORE", str(tmp_path))

    loaded = _load_hrrr()

    assert loaded.shape == (1, 1, 100, 100)
    assert loaded.dtype == torch.float32
    assert torch.equal(loaded[0], values[1].to(torch.float32))


def test_load_hrrr_crops_center_without_changing_sample_filter(tmp_path, monkeypatch):
    hrrr_dir = tmp_path / "hrrr_data"
    hrrr_dir.mkdir()
    values = torch.arange(2 * 100 * 100, dtype=torch.float32).reshape(2, 1, 100, 100)
    values[0, 0, 0, 0] = float("nan")
    torch.save(values, hrrr_dir / "hrrr_apcp_100x100.pt")
    monkeypatch.setenv("STORE", str(tmp_path))

    loaded = _load_hrrr(crop_size=64)

    assert loaded.shape == (1, 1, 64, 64)
    assert torch.equal(loaded[0], values[1, :, 18:82, 18:82])
    with pytest.raises(ValueError, match="crop_size"):
        _load_hrrr(crop_size=101)


def test_load_hrrr_accepts_saved_payload_dict(tmp_path, monkeypatch):
    hrrr_dir = tmp_path / "hrrr_data"
    hrrr_dir.mkdir()
    values = torch.arange(3 * 1 * 100 * 100, dtype=torch.float64).reshape(3, 1, 100, 100)
    torch.save({"timestamps": ["t0", "t1", "t2"], "frames": values}, hrrr_dir / "hrrr_apcp_100x100.pt")
    monkeypatch.setenv("STORE", str(tmp_path))

    loaded = _load_hrrr()

    assert loaded.shape == (3, 1, 100, 100)
    assert loaded.dtype == torch.float32
    assert torch.equal(loaded, values.to(torch.float32))


def test_load_hrrr_accepts_apcp_mm_payload_key(tmp_path, monkeypatch):
    hrrr_dir = tmp_path / "hrrr_data"
    hrrr_dir.mkdir()
    values = torch.arange(2 * 1 * 100 * 100, dtype=torch.float64).reshape(2, 1, 100, 100)
    torch.save({"apcp_mm": values, "variable": "APCP"}, hrrr_dir / "hrrr_apcp_100x100.pt")
    monkeypatch.setenv("STORE", str(tmp_path))

    loaded = _load_hrrr()

    assert loaded.shape == (2, 1, 100, 100)
    assert loaded.dtype == torch.float32
    assert torch.equal(loaded, values.to(torch.float32))


def test_load_hrrr_rejects_unknown_payload_dict(tmp_path, monkeypatch):
    hrrr_dir = tmp_path / "hrrr_data"
    hrrr_dir.mkdir()
    torch.save({"timestamps": ["t0"]}, hrrr_dir / "hrrr_apcp_100x100.pt")
    monkeypatch.setenv("STORE", str(tmp_path))

    with pytest.raises(ValueError, match="Expected a tensor payload or a dict containing 'frames' or 'apcp_mm'"):
        _load_hrrr()


def test_load_hrrr_raises_when_fixed_file_is_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("STORE", str(tmp_path))

    with pytest.raises(FileNotFoundError, match="hrrr_data/hrrr_apcp_100x100.pt"):
        _load_hrrr()


def test_load_hrrr_rejects_loader_kwargs(tmp_path, monkeypatch):
    hrrr_dir = tmp_path / "hrrr_data"
    hrrr_dir.mkdir()
    torch.save(torch.ones(1, 1, 100, 100), hrrr_dir / "hrrr_apcp_100x100.pt")
    monkeypatch.setenv("STORE", str(tmp_path))

    with pytest.raises(TypeError, match="Unexpected HRRR loader kwargs: n_workers"):
        _load_hrrr(n_workers=8)


def test_fetch_real_data_supports_hrrr_tensor_dataset(tmp_path, monkeypatch):
    hrrr_dir = tmp_path / "hrrr_data"
    hrrr_dir.mkdir()
    torch.save(
        torch.arange(5 * 1 * 100 * 100, dtype=torch.float32).reshape(5, 1, 100, 100),
        hrrr_dir / "hrrr_apcp_100x100.pt",
    )
    monkeypatch.setenv("STORE", str(tmp_path))

    x_train, x_val, x_test = fetch_real_data(
        "hrrr",
        val_size=0.2,
        test_size=0.2,
        random_state=0,
        dtype=torch.float64,
    )

    assert x_train.shape == (3, 1, 100, 100)
    assert x_val.shape == (1, 1, 100, 100)
    assert x_test.shape == (1, 1, 100, 100)
    assert x_train.dtype == torch.float64
    assert x_val.dtype == torch.float64
    assert x_test.dtype == torch.float64


def test_hrrr_dataset_is_registered():
    assert "hrrr" in list_real_datasets()
    assert get_real_dataset_metadata("hrrr") == {
        "name": "hrrr",
        "tail_index_alpha": None,
        "description": "HRRR accumulated precipitation fields on a 100x100 crop.",
        "split_mode": "random",
        "dataset_type": "real",
        "dim": (1, 100, 100),
        "n_samples": None,
    }


def test_load_lvis_reads_local_fixture_and_filters_frequency(tmp_path):
    data_home = _write_lvis_fixture(tmp_path / "lvis")

    loaded = _load_lvis(
        data_home=data_home,
        split="train",
        image_size=8,
        max_samples=10,
        category_frequency="r",
    )

    assert isinstance(loaded, DatasetPayload)
    tensor = loaded.data
    assert tensor.shape == (3, 3, 8, 8)
    assert tensor.dtype == torch.float32
    assert tensor.min().item() >= 0.0
    assert tensor.max().item() <= 1.0
    assert loaded.metadata["source_split"] == "train"
    assert loaded.metadata["category_frequency_filter"] == "r"
    assert loaded.metadata["n_selected_images"] == 3
    assert [record["image_id"] for record in loaded.metadata["records"]] == [1, 3, 5]
    assert all(record["category_frequencies"] == ["r"] for record in loaded.metadata["records"])


def test_load_lvis_reuses_processed_cache(tmp_path, monkeypatch):
    import jeanzaydata._dataset as dataset_module

    data_home = _write_lvis_fixture(tmp_path / "lvis")
    cache_dir = tmp_path / "cache"

    first = _load_lvis(
        data_home=data_home,
        cache_dir=cache_dir,
        split="train",
        image_size=8,
        max_samples=3,
        seed=0,
    )

    def fail_read(*args, **kwargs):
        raise AssertionError("cache miss")

    monkeypatch.setattr(dataset_module, "read_rgb_resized", fail_read)
    second = _load_lvis(
        data_home=data_home,
        cache_dir=cache_dir,
        split="train",
        image_size=8,
        max_samples=3,
        seed=0,
    )

    assert second.metadata["cache_hit"] is True
    assert second.metadata["cache_path"] == first.metadata["cache_path"]
    assert torch.equal(second.data, first.data)


def test_fetch_real_data_supports_lvis_tensor_dataset(tmp_path):
    data_home = _write_lvis_fixture(tmp_path / "lvis")

    x_train, x_val, x_test = fetch_real_data(
        "lvis",
        data_home=data_home,
        split="train",
        image_size=8,
        max_samples=5,
        val_size=0.2,
        test_size=0.2,
        random_state=0,
        standardize=False,
    )

    assert x_train.shape == (3, 3, 8, 8)
    assert x_val.shape == (1, 3, 8, 8)
    assert x_test.shape == (1, 3, 8, 8)
    assert x_train.dtype == torch.float32


def test_fetch_real_data_passes_lvis_n_samples_as_max_samples(tmp_path):
    data_home = _write_lvis_fixture(tmp_path / "lvis")

    x_train, x_val, x_test = fetch_real_data(
        "lvis",
        data_home=data_home,
        split="train",
        image_size=8,
        n_samples=5,
        val_size=0.2,
        test_size=0.2,
        random_state=0,
        standardize=False,
    )

    assert x_train.shape[0] + x_val.shape[0] + x_test.shape[0] == 5


def test_fetch_real_data_return_metadata_includes_lvis_records_and_histograms(tmp_path):
    data_home = _write_lvis_fixture(tmp_path / "lvis")

    x_train, x_val, x_test, metadata = fetch_real_data(
        "lvis",
        data_home=data_home,
        split="train",
        image_size=8,
        max_samples=5,
        val_size=0.2,
        test_size=0.2,
        random_state=0,
        standardize=False,
        return_metadata=True,
    )

    assert x_train.shape == (3, 3, 8, 8)
    assert x_val.shape == (1, 3, 8, 8)
    assert x_test.shape == (1, 3, 8, 8)
    assert metadata["dataset"] == get_real_dataset_metadata("lvis")
    assert metadata["request"]["name"] == "lvis"
    assert metadata["request"]["params"]["max_samples"] == 5
    assert metadata["loader"]["image_size"] == 8
    assert metadata["loader"]["n_selected_images"] == 5

    split_records = [
        record
        for split_name in ("train", "val", "test")
        for record in metadata["splits"][split_name]["records"]
    ]
    assert len(split_records) == 5
    assert len({record["image_id"] for record in split_records}) == 5
    assert sum(metadata["splits"][name]["n_samples"] for name in ("train", "val", "test")) == 5
    assert all("category_histogram" in metadata["splits"][name] for name in ("train", "val", "test"))
    assert all("frequency_histogram" in metadata["splits"][name] for name in ("train", "val", "test"))


def test_lvis_uses_dsdir_and_fails_clearly_when_absent(monkeypatch):
    monkeypatch.delenv("DSDIR", raising=False)

    with pytest.raises(RuntimeError, match="DSDIR"):
        _load_lvis(max_samples=1)


def test_lvis_dataset_is_registered():
    assert "lvis" in list_real_datasets()
    assert get_real_dataset_metadata("lvis") == {
        "name": "lvis",
        "tail_index_alpha": None,
        "description": "LVIS long-tailed object categories from local Jean Zay COCO/LVIS files; default image_size=64.",
        "split_mode": "random",
        "dataset_type": "real",
        "dim": (3, 64, 64),
        "n_samples": None,
    }


def test_lvis_smoke_from_dsdir_if_available():
    if not os.getenv("DSDIR"):
        pytest.skip("Jean Zay $DSDIR is not set.")
    try:
        loaded = _load_lvis(split="train", image_size=8, max_samples=3, seed=0)
    except RuntimeError as exc:
        pytest.skip(f"LVIS assets are not available under $DSDIR: {exc}")

    assert isinstance(loaded, DatasetPayload)
    tensor = loaded.data
    assert tensor.shape == (3, 3, 8, 8)
    assert tensor.dtype == torch.float32
    assert tensor.min().item() >= 0.0
    assert tensor.max().item() <= 1.0
    assert len(loaded.metadata["records"]) == 3


def test_resolve_real_data_home_jeanzay_data_home_takes_precedence(tmp_path, monkeypatch):
    custom = tmp_path / "custom_root"
    monkeypatch.delenv("JEANZAY_DATA", raising=False)
    monkeypatch.setenv("JEANZAY_DATA_HOME", str(custom))
    monkeypatch.setenv("WORK", str(tmp_path / "work"))
    resolved = _resolve_real_data_home()
    assert resolved == custom
    assert custom.exists()


def test_resolve_real_data_home_jeanzay_data_takes_top_precedence(tmp_path, monkeypatch):
    primary = tmp_path / "primary_root"
    monkeypatch.setenv("JEANZAY_DATA", str(primary))
    monkeypatch.setenv("JEANZAY_DATA_HOME", str(tmp_path / "secondary"))
    monkeypatch.setenv("WORK", str(tmp_path / "work"))
    resolved = _resolve_real_data_home()
    assert resolved == primary
    assert primary.exists()


def test_split_sample_indices_chronological_preserves_temporal_order():
    train, val, test = split_sample_indices(
        20, val_size=0.2, test_size=0.2, random_state=0, split_mode="chronological"
    )
    assert int(train[-1]) < int(val[0])
    assert int(val[-1]) < int(test[0])


def test_split_sample_indices_chronological_too_small_raises():
    # 3 rows with val/test=0.2 each: floor(0.2*3)=0 -> empty val/test split
    with pytest.raises(ValueError, match="empty split"):
        split_sample_indices(3, val_size=0.2, test_size=0.2, random_state=0, split_mode="chronological")


def test_split_sample_indices_invalid_split_mode_raises():
    with pytest.raises(ValueError, match="split_mode"):
        split_sample_indices(10, val_size=0.2, test_size=0.2, random_state=0, split_mode="bad_mode")


def test_standardize_split_arrays_zero_variance_column_does_not_produce_nan():
    train = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    val = np.array([[1.0, 3.0]])
    test = np.array([[1.0, -1.0]])
    tr, v, te = _standardize_split_arrays(train, val, test)
    assert np.isfinite(tr).all()
    assert np.isfinite(v).all()
    assert np.isfinite(te).all()
    # constant column → mean=1, std clamped to 1 → (1-1)/1 = 0
    assert (tr[:, 0] == 0.0).all()


def test_resolve_dataset_type_mismatch_raises_value_error():
    fake_datasets = {"gaussian": SimpleNamespace(dataset_type="synthetic")}
    with pytest.raises(ValueError, match="expected 'real'"):
        _resolve_dataset("gaussian", fake_datasets, dataset_type="real")


def test_fetch_real_data_n_samples_exceeding_available_warns(monkeypatch):
    small_df = pd.DataFrame({"x": list(range(6)), "y": list(range(6, 12))})
    fake_entry = SimpleNamespace(
        loader=lambda **kw: small_df,
        split_mode="random",
        dataset_type="real",
        standardize_default=False,
    )
    monkeypatch.setattr("jeanzaydata._dataset._resolve_dataset", lambda *a, **kw: fake_entry)
    with pytest.warns(UserWarning, match="exceeds available"):
        fetch_real_data("anything", n_samples=100, val_size=0.2, test_size=0.2)


# ---------------------------------------------------------------------------
# CIFAR-100-LT
# ---------------------------------------------------------------------------


def _write_cifar100_fixture(root, *, samples_per_class=4, n_classes=6, image_hw=32, nested=True):
    base = root / "cifar-100-python" if nested else root
    base.mkdir(parents=True)
    rng = np.random.default_rng(123)
    n_train = samples_per_class * n_classes
    train_data = rng.integers(0, 256, size=(n_train, 3 * image_hw * image_hw), dtype=np.uint8)
    train_labels = np.repeat(np.arange(n_classes, dtype=np.int64), samples_per_class).tolist()
    n_test = max(1, samples_per_class // 2) * n_classes
    test_data = rng.integers(0, 256, size=(n_test, 3 * image_hw * image_hw), dtype=np.uint8)
    test_labels = np.repeat(np.arange(n_classes, dtype=np.int64), max(1, samples_per_class // 2)).tolist()

    train_payload = {
        b"data": train_data,
        b"fine_labels": train_labels,
        b"coarse_labels": [0] * n_train,
        b"batch_label": b"training batch 1 of 1",
        b"filenames": [f"train_{i}.png".encode("utf-8") for i in range(n_train)],
    }
    test_payload = {
        b"data": test_data,
        b"fine_labels": test_labels,
        b"coarse_labels": [0] * n_test,
        b"batch_label": b"testing batch 1 of 1",
        b"filenames": [f"test_{i}.png".encode("utf-8") for i in range(n_test)],
    }
    meta_payload = {
        b"fine_label_names": [f"class_{c}".encode("utf-8") for c in range(n_classes)],
        b"coarse_label_names": [b"super_0"],
    }
    with (base / "train").open("wb") as handle:
        pickle.dump(train_payload, handle)
    with (base / "test").open("wb") as handle:
        pickle.dump(test_payload, handle)
    with (base / "meta").open("wb") as handle:
        pickle.dump(meta_payload, handle)
    return root


def test_load_cifar100_lt_reads_local_fixture_and_long_tails_classes(tmp_path):
    root = _write_cifar100_fixture(tmp_path / "cifar100_lt", samples_per_class=8, n_classes=4)

    loaded = _load_cifar100_lt(
        data_home=root,
        split="train",
        image_size=8,
        imbalance_factor=8,
        cache=False,
    )

    assert isinstance(loaded, DatasetPayload)
    assert loaded.data.dtype == torch.float32
    assert loaded.data.shape[1:] == (3, 8, 8)
    assert loaded.data.min().item() >= 0.0
    assert loaded.data.max().item() <= 1.0
    counts = loaded.metadata["long_tail_class_counts"]
    assert counts == [8, 4, 2, 1]
    assert loaded.data.shape[0] == sum(counts)
    histogram = loaded.metadata["class_histogram"]
    assert {int(k): int(v) for k, v in histogram.items()} == {0: 8, 1: 4, 2: 2, 3: 1}
    assert loaded.metadata["label_names"] == ["class_0", "class_1", "class_2", "class_3"]


def test_load_cifar100_lt_max_samples_caps_total(tmp_path):
    root = _write_cifar100_fixture(tmp_path / "cifar100_lt", samples_per_class=8, n_classes=4)

    loaded = _load_cifar100_lt(
        data_home=root,
        split="train",
        image_size=8,
        imbalance_factor=8,
        max_samples=5,
        cache=False,
    )

    assert loaded.data.shape[0] == 5
    assert len(loaded.metadata["records"]) == 5


def test_load_cifar100_lt_is_deterministic_under_seed(tmp_path):
    root = _write_cifar100_fixture(tmp_path / "cifar100_lt", samples_per_class=8, n_classes=4)

    first = _load_cifar100_lt(data_home=root, split="train", image_size=8, imbalance_factor=8, seed=7, cache=False)
    second = _load_cifar100_lt(data_home=root, split="train", image_size=8, imbalance_factor=8, seed=7, cache=False)
    third = _load_cifar100_lt(data_home=root, split="train", image_size=8, imbalance_factor=8, seed=11, cache=False)

    assert first.metadata["selected_indices"] == second.metadata["selected_indices"]
    assert first.metadata["selected_indices"] != third.metadata["selected_indices"]
    assert torch.equal(first.data, second.data)


def test_load_cifar100_lt_reuses_processed_cache(tmp_path, monkeypatch):
    import jeanzaydata._dataset as dataset_module

    root = _write_cifar100_fixture(tmp_path / "cifar100_lt", samples_per_class=8, n_classes=4)
    cache_dir = tmp_path / "cache"

    first = _load_cifar100_lt(
        data_home=root,
        cache_dir=cache_dir,
        split="train",
        image_size=8,
        imbalance_factor=8,
        seed=0,
    )

    monkeypatch.setattr(dataset_module, "load_cifar100_lt_arrays",
                        lambda **kwargs: (_ for _ in ()).throw(AssertionError("cache miss")))
    second = _load_cifar100_lt(
        data_home=root,
        cache_dir=cache_dir,
        split="train",
        image_size=8,
        imbalance_factor=8,
        seed=0,
    )

    assert second.metadata["cache_hit"] is True
    assert second.metadata["cache_path"] == first.metadata["cache_path"]
    assert torch.equal(second.data, first.data)


def test_load_cifar100_lt_flat_layout_without_subdir(tmp_path):
    root = _write_cifar100_fixture(tmp_path / "cifar100_lt", samples_per_class=4, n_classes=3, nested=False)

    loaded = _load_cifar100_lt(
        data_home=root,
        split="test",
        image_size=8,
        imbalance_factor=4,
        cache=False,
    )

    assert loaded.data.shape[0] == sum(loaded.metadata["long_tail_class_counts"])


def test_cifar100_lt_raises_actionable_error_when_no_root_configured(monkeypatch):
    monkeypatch.delenv("JEANZAY_DATA", raising=False)

    with pytest.raises(RuntimeError, match="JEANZAY_DATA"):
        _load_cifar100_lt(cache=False)


def test_cifar100_lt_raises_when_root_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("JEANZAY_DATA", str(tmp_path / "missing"))

    with pytest.raises(RuntimeError, match="root does not exist"):
        _load_cifar100_lt(cache=False)


def test_cifar100_lt_raises_when_pickle_files_missing(tmp_path):
    root = tmp_path / "cifar100_lt"
    root.mkdir()

    with pytest.raises(RuntimeError, match="not found under"):
        _load_cifar100_lt(data_home=root, cache=False)


def test_cifar100_lt_resolves_under_jeanzay_data_env(tmp_path, monkeypatch):
    data_root = tmp_path / "jz_datasets"
    cifar_root = data_root / "raw" / "cifar100_lt"
    _write_cifar100_fixture(cifar_root, samples_per_class=4, n_classes=3)
    monkeypatch.setenv("JEANZAY_DATA", str(data_root))

    loaded = _load_cifar100_lt(split="train", image_size=8, imbalance_factor=4, cache=False)

    assert loaded.data.shape[0] == sum(loaded.metadata["long_tail_class_counts"])
    assert str(cifar_root) in loaded.metadata["source_path"]


def test_cifar100_lt_is_registered():
    assert "cifar100_lt" in list_real_datasets()
    assert get_real_dataset_metadata("cifar100_lt") == {
        "name": "cifar100_lt",
        "tail_index_alpha": None,
        "description": "CIFAR-100 reshaped into a long-tailed subset using exponential class decay; default image_size=64.",
        "split_mode": "random",
        "dataset_type": "real",
        "dim": (3, 64, 64),
        "n_samples": None,
    }


def test_fetch_real_data_supports_cifar100_lt(tmp_path):
    root = _write_cifar100_fixture(tmp_path / "cifar100_lt", samples_per_class=8, n_classes=4)

    x_train, x_val, x_test, metadata = fetch_real_data(
        "cifar100_lt",
        data_home=root,
        split="train",
        image_size=8,
        imbalance_factor=8,
        val_size=0.2,
        test_size=0.2,
        random_state=0,
        standardize=False,
        return_metadata=True,
    )

    total = x_train.shape[0] + x_val.shape[0] + x_test.shape[0]
    assert total == sum([8, 4, 2, 1])
    assert x_train.dtype == torch.float32
    assert x_train.shape[1:] == (3, 8, 8)
    assert metadata["loader"]["imbalance_factor"] == 8.0
    assert metadata["loader"]["image_size"] == 8
    assert metadata["request"]["name"] == "cifar100_lt"


def test_fetch_real_data_passes_cifar100_lt_n_samples_as_max_samples(tmp_path):
    root = _write_cifar100_fixture(tmp_path / "cifar100_lt", samples_per_class=8, n_classes=4)

    x_train, x_val, x_test = fetch_real_data(
        "cifar100_lt",
        data_home=root,
        split="train",
        image_size=8,
        imbalance_factor=8,
        n_samples=4,
        val_size=0.25,
        test_size=0.25,
        random_state=0,
        standardize=False,
    )

    assert x_train.shape[0] + x_val.shape[0] + x_test.shape[0] == 4


# ---------------------------------------------------------------------------
# ImageNet-LT
# ---------------------------------------------------------------------------


def _write_imagenet_lt_fixture(root, *, with_imagenet_subdir=True):
    image_mod = pytest.importorskip("PIL.Image")
    annotations_dir = root / "annotations"
    annotations_dir.mkdir(parents=True)
    if with_imagenet_subdir:
        image_root = root / "imagenet"
    else:
        image_root = root / "images"
    image_root.mkdir(parents=True)

    records = [
        ("train/n00000001/img1.JPEG", 0, (255, 0, 0)),
        ("train/n00000001/img2.JPEG", 0, (220, 0, 0)),
        ("train/n00000001/img3.JPEG", 0, (190, 0, 0)),
        ("train/n00000002/img4.JPEG", 1, (0, 255, 0)),
        ("train/n00000002/img5.JPEG", 1, (0, 220, 0)),
        ("train/n00000003/img6.JPEG", 2, (0, 0, 255)),
    ]
    lines = []
    for relative_path, class_id, color in records:
        path = image_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        image_mod.new("RGB", (12, 14), color=color).save(path)
        lines.append(f"{relative_path} {class_id}")
    (annotations_dir / "ImageNet_LT_train.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root, image_root


def test_load_imagenet_lt_reads_local_fixture(tmp_path):
    root, image_root = _write_imagenet_lt_fixture(tmp_path / "imagenet_lt")

    loaded = _load_imagenet_lt(
        data_home=root,
        imagenet_root=image_root,
        split="train",
        image_size=8,
        cache=False,
    )

    assert isinstance(loaded, DatasetPayload)
    assert loaded.data.shape == (6, 3, 8, 8)
    assert loaded.data.dtype == torch.float32
    assert loaded.data.min().item() >= 0.0
    assert loaded.data.max().item() <= 1.0
    assert loaded.metadata["source_split"] == "train"
    assert loaded.metadata["n_selected_images"] == 6
    assert loaded.metadata["class_histogram"] == {0: 3, 1: 2, 2: 1}
    assert [record["wnid"] for record in loaded.metadata["records"]] == [
        "n00000001", "n00000001", "n00000001", "n00000002", "n00000002", "n00000003",
    ]


def test_load_imagenet_lt_max_samples_is_deterministic(tmp_path):
    root, image_root = _write_imagenet_lt_fixture(tmp_path / "imagenet_lt")

    first = _load_imagenet_lt(
        data_home=root, imagenet_root=image_root, split="train",
        image_size=8, max_samples=3, seed=5, cache=False,
    )
    second = _load_imagenet_lt(
        data_home=root, imagenet_root=image_root, split="train",
        image_size=8, max_samples=3, seed=5, cache=False,
    )
    other = _load_imagenet_lt(
        data_home=root, imagenet_root=image_root, split="train",
        image_size=8, max_samples=3, seed=99, cache=False,
    )

    assert first.metadata["n_selected_images"] == 3
    assert [r["relative_path"] for r in first.metadata["records"]] == [r["relative_path"] for r in second.metadata["records"]]
    assert torch.equal(first.data, second.data)
    assert [r["relative_path"] for r in first.metadata["records"]] != [r["relative_path"] for r in other.metadata["records"]]


def test_load_imagenet_lt_reuses_processed_cache(tmp_path, monkeypatch):
    import jeanzaydata._dataset as dataset_module

    root, image_root = _write_imagenet_lt_fixture(tmp_path / "imagenet_lt")
    cache_dir = tmp_path / "cache"

    first = _load_imagenet_lt(
        data_home=root, imagenet_root=image_root, cache_dir=cache_dir,
        split="train", image_size=8, max_samples=3, seed=0,
    )

    def fail_read(*args, **kwargs):
        raise AssertionError("cache miss")

    monkeypatch.setattr(dataset_module, "read_rgb_resized", fail_read)
    second = _load_imagenet_lt(
        data_home=root, imagenet_root=image_root, cache_dir=cache_dir,
        split="train", image_size=8, max_samples=3, seed=0,
    )

    assert second.metadata["cache_hit"] is True
    assert second.metadata["cache_path"] == first.metadata["cache_path"]
    assert torch.equal(second.data, first.data)


def test_load_imagenet_lt_uses_explicit_annotation_path(tmp_path):
    root, image_root = _write_imagenet_lt_fixture(tmp_path / "imagenet_lt")
    elsewhere = tmp_path / "elsewhere" / "annot.txt"
    elsewhere.parent.mkdir()
    elsewhere.write_text((root / "annotations" / "ImageNet_LT_train.txt").read_text(), encoding="utf-8")

    loaded = _load_imagenet_lt(
        annotation_path=elsewhere,
        imagenet_root=image_root,
        split="train",
        image_size=8,
        cache=False,
    )
    assert loaded.metadata["annotation_path"] == str(elsewhere)
    assert loaded.data.shape[0] == 6


def test_imagenet_lt_raises_when_data_home_missing(monkeypatch):
    monkeypatch.delenv("JEANZAY_DATA", raising=False)

    with pytest.raises(RuntimeError, match="JEANZAY_DATA"):
        _load_imagenet_lt(cache=False)


def test_imagenet_lt_raises_when_annotation_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("JEANZAY_DATA", str(tmp_path))
    raw_root = tmp_path / "raw" / "imagenet_lt" / "annotations"
    raw_root.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="annotation file not found"):
        _load_imagenet_lt(cache=False)


def test_imagenet_lt_raises_when_image_missing(tmp_path):
    root, image_root = _write_imagenet_lt_fixture(tmp_path / "imagenet_lt")
    (image_root / "train" / "n00000001" / "img1.JPEG").unlink()

    with pytest.raises(RuntimeError, match="image not found"):
        _load_imagenet_lt(
            data_home=root, imagenet_root=image_root,
            split="train", image_size=8, cache=False,
        )


def test_imagenet_lt_resolves_imagenet_root_via_data_home_subdir(tmp_path):
    root, image_root = _write_imagenet_lt_fixture(tmp_path / "imagenet_lt", with_imagenet_subdir=True)

    loaded = _load_imagenet_lt(
        data_home=root,
        split="train",
        image_size=8,
        cache=False,
    )
    assert loaded.metadata["imagenet_root"] == str(image_root)
    assert loaded.data.shape[0] == 6


def test_imagenet_lt_is_registered():
    assert "imagenet_lt" in list_real_datasets()
    assert get_real_dataset_metadata("imagenet_lt") == {
        "name": "imagenet_lt",
        "tail_index_alpha": None,
        "description": "ImageNet-LT split using shipped annotation files and a local ImageNet image tree; default image_size=64.",
        "split_mode": "random",
        "dataset_type": "real",
        "dim": (3, 64, 64),
        "n_samples": None,
    }


def test_fetch_real_data_supports_imagenet_lt(tmp_path):
    root, image_root = _write_imagenet_lt_fixture(tmp_path / "imagenet_lt")

    x_train, x_val, x_test, metadata = fetch_real_data(
        "imagenet_lt",
        data_home=root,
        imagenet_root=image_root,
        split="train",
        image_size=8,
        max_samples=4,
        val_size=0.25,
        test_size=0.25,
        random_state=0,
        standardize=False,
        return_metadata=True,
    )
    assert x_train.shape == (2, 3, 8, 8)
    assert x_val.shape == (1, 3, 8, 8)
    assert x_test.shape == (1, 3, 8, 8)
    assert metadata["request"]["name"] == "imagenet_lt"
    assert metadata["loader"]["n_selected_images"] == 4


def test_fetch_real_data_passes_imagenet_lt_n_samples_as_max_samples(tmp_path):
    root, image_root = _write_imagenet_lt_fixture(tmp_path / "imagenet_lt")

    x_train, x_val, x_test = fetch_real_data(
        "imagenet_lt",
        data_home=root,
        imagenet_root=image_root,
        split="train",
        image_size=8,
        n_samples=3,
        val_size=0.25,
        test_size=0.25,
        random_state=0,
        standardize=False,
    )
    assert x_train.shape[0] + x_val.shape[0] + x_test.shape[0] == 3
