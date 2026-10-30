"""Image transforms used when loading Oxford-IIIT Pet samples."""
from __future__ import annotations

from collections.abc import Callable

from torch.utils.data import Dataset

from train.hyperparameters import DATA_PARAMS


class TransformedDataset(Dataset):
    """Apply an image transform when a sample is fetched from a dataset."""

    def __init__(self, dataset, transform: Callable):
        """Wrap a dataset with a transform applied to each fetched image.

        :param dataset, Dataset; Source dataset returning Hugging Face image/label rows.
        :param transform, Callable; Image transform to apply on sample access.
        :return: None.
        """
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        """Return the number of samples in the wrapped dataset.

        :return: Number of samples.
        """
        return len(self.dataset)

    def __getitem__(self, index):
        """Fetch a sample and transform its image.

        :param index, int; Index of the sample to fetch.
        :return: Transformed image and its original target.
        """
        sample = self.dataset[index]
        image = sample["image"].convert("RGB")
        return self.transform(image), int(sample["label"])


def build_train_transform():
    """Create random augmentation and normalization for training images.

    :return: Composition of random image augmentation and normalization transforms.
    """
    from torchvision import transforms as T

    return T.Compose(
        [
            T.RandomResizedCrop(
                DATA_PARAMS["image_size"], scale=DATA_PARAMS["train_crop_scale"]
            ),
            T.RandomHorizontalFlip(
                p=DATA_PARAMS["train_horizontal_flip_prob"]
            ),
            T.ToTensor(),
            T.Normalize(DATA_PARAMS["imagenet_mean"], DATA_PARAMS["imagenet_std"]),
        ]
    )


def build_eval_transform():
    """Create deterministic transforms for validation and test images.

    :return: Composition of resize, center-crop, tensor, and normalization transforms.
    """
    from torchvision import transforms as T

    return T.Compose(
        [
            T.Resize(
                round(DATA_PARAMS["image_size"] * DATA_PARAMS["eval_resize_scale"])
            ),
            T.CenterCrop(DATA_PARAMS["image_size"]),
            T.ToTensor(),
            T.Normalize(DATA_PARAMS["imagenet_mean"], DATA_PARAMS["imagenet_std"]),
        ]
    )
