from typing import Callable

from torch.utils.data import Dataset
from datasets import load_dataset


def get_hf_dataset(
    ds_name: str,
    ds_cache_dir: str,
    tokenize_fn: Callable | None = None,
    mapping_kwargs: dict = {},
    filter_fn: Callable | None = None,
    format: str | None = None,
) -> dict[str, Dataset]:
    dataset = load_dataset(
        ds_name,
        cache_dir=ds_cache_dir,
    )
    train_ds = dataset["train"]
    val_ds = dataset["validation"]

    # apply the tokenize function to the dataset if provided
    # we can use the mapping_kwargs to specify the arguments 
    # for the map function (e.g., batched, batch_size, remove_columns, etc.)
    if tokenize_fn is not None:
        train_ds = train_ds.map(tokenize_fn, **mapping_kwargs)
        val_ds = val_ds.map(tokenize_fn, **mapping_kwargs)

    ### filter out samples based on the provided filter function
    if filter_fn is not None:
        train_ds = train_ds.filter(filter_fn)
        val_ds = val_ds.filter(filter_fn)

    ### set dataset format tensors (e.g., PyTorch tensors)
    if format is not None:
        train_ds.set_format(format)
        val_ds.set_format(format)

    return {"train": train_ds, "validation": val_ds}

