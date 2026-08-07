"""Tests for Jean Zay raw-data staging helpers."""

import os
import subprocess
import sys
from jeanzaydata._prepare import prepare_cifar100_lt, prepare_imagenet_lt


def test_prepare_cifar100_lt_symlinks_source(tmp_path):
    source = tmp_path / "source" / "cifar-100-python"
    source.mkdir(parents=True)
    for name in ("train", "test", "meta"):
        (source / name).write_bytes(b"fixture")

    root = tmp_path / "root"
    prepare_cifar100_lt(root=root, source=source)

    assert (root / "cifar-100-python" / "train").is_file()


def test_prepare_imagenet_lt_copies_annotations_and_symlinks_source(tmp_path):
    source = tmp_path / "imagenet"
    source.mkdir()
    root = tmp_path / "imagenet_lt"

    prepare_imagenet_lt(root=root, source=source)

    assert (root / "annotations" / "ImageNet_LT_train.txt").is_file()
    assert (root / "imagenet").exists()


def test_jeanzaydata_module_cli_does_not_import_torch():
    code = (
        "import builtins, runpy, sys\n"
        "_orig_import = builtins.__import__\n"
        "def guard(name, *args, **kwargs):\n"
        "    if name == 'torch' or name.startswith('torch.'):\n"
        "        raise RuntimeError(f'torch import attempted via {name!r}')\n"
        "    return _orig_import(name, *args, **kwargs)\n"
        "builtins.__import__ = guard\n"
        "sys.argv = ['jeanzaydata', '--help']\n"
        "runpy.run_module('jeanzaydata', run_name='__main__')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stdout + result.stderr
