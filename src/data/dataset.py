"""Load a Hugging Face dataset and create a stratified train/validation split."""


def build_datasets(dataset_name: str, val_frac: float, seed: int):
    """Load the dataset and split its training set while preserving class ratios.

    Hugging Face caches the download automatically. The official test split is
    kept separate and unchanged.

    :param val_frac, float; Fraction of training samples assigned to validation.
    :param seed, int; Seed used for a reproducible stratified split.
    :param dataset_name, str; Hugging Face dataset repository identifier.
    :return: Train, validation, and test splits, followed by dataset metadata.
    """
    from datasets import load_dataset

    dataset = load_dataset(dataset_name)
    splits = dataset["train"].train_test_split(
        test_size=val_frac,
        seed=seed,
        stratify_by_column="label",
    )
    classes = dataset["train"].features["label"].names
    info = {
        "num_classes": len(classes),
        "classes": classes,
        "n_train": len(splits["train"]),
        "n_val": len(splits["test"]),
        "n_test": len(dataset["test"]),
    }
    return splits["train"], splits["test"], dataset["test"], info
