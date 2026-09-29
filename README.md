# JeanZayData

Jean Zay dataset staging and loading for LVIS, CIFAR-100-LT, ImageNet-LT, and HRRR. It provides reproducible splits and uses the site's `DSDIR`, `WORK`, and `STORE` paths.

```bash
pip install "jeanzaydata @ git+https://github.com/Diffusion-Research-Lab/jeanzaydata.git"
```

For local development:

```bash
git clone git@github.com:Diffusion-Research-Lab/jeanzaydata.git
cd jeanzaydata
python -m pip install -e ".[dev]"
```

Set the writable cache root, then stage shared datasets as needed:

```bash
export JEANZAY_DATA="$WORK/jz_datasets"
jeanzaydata init cifar100_lt
jeanzaydata init imagenet_lt
```

Load a dataset with:

```python
from jeanzaydata import fetch_real_data

x_train, x_val, x_test, metadata = fetch_real_data(
    "cifar100_lt",
    image_size=64,
    imbalance_factor=10,
    return_metadata=True,
)
```

Run local checks with:

```bash
make setup
make check
```
