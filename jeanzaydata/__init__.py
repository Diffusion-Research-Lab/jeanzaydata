"""Jean Zay dataset loading and staging package."""

__all__ = ["fetch_real_data", "get_dataset_metadata", "list_datasets"]


def fetch_real_data(target_data, **kwargs):
    from ._dataset import fetch_real_data as _fetch_real_data
    return _fetch_real_data(target_data, **kwargs)


def get_dataset_metadata(target_data):
    from ._dataset import get_dataset_metadata as _get_dataset_metadata
    return _get_dataset_metadata(target_data)


def list_datasets():
    from ._dataset import list_datasets as _list_datasets
    return _list_datasets()
