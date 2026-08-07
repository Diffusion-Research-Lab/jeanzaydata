# JeanZayData

JeanZayData stages and loads real datasets from the filesystem of the Jean Zay supercomputer. It provides reproducible train/validation/test splits, processed tensor caches, and metadata for LVIS, CIFAR-100-LT, ImageNet-LT, and HRRR precipitation fields.

The package is intentionally site-specific. Its defaults use Jean Zay's shared dataset trees and the standard `DSDIR`, `WORK`, and `STORE` filesystem variables.

## Installation

```bash
pip install "jeanzaydata @ git+ssh://git@github.com/Diffusion-Research-Lab/jeanzaydata.git"
```

For development:

```bash
git clone git@github.com:Diffusion-Research-Lab/jeanzaydata.git
cd jeanzaydata
python -m pip install -e ".[dev]"
```

## Dataset root

Set `JEANZAY_DATA` to the package's writable staging and cache root.

```bash
export JEANZAY_DATA="$WORK/jz_datasets"
```

## Staging shared datasets

The CLI creates a writable dataset layout while linking to Jean Zay's shared CIFAR-100 and ImageNet trees:

```bash
jeanzaydata init cifar100_lt
jeanzaydata init imagenet_lt
```

Equivalent module invocation:

```bash
python -m jeanzaydata init imagenet_lt
```

## Loading datasets

```python
from jeanzaydata import fetch_real_data

x_train, x_val, x_test, metadata = fetch_real_data(
    "cifar100_lt",
    image_size=64,
    imbalance_factor=10,
    return_metadata=True,
)
```

## Development

```bash
make setup
make check
```

## License

JeanZayData is released under the MIT License.
