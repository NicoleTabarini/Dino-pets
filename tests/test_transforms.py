import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

import torch
from PIL import Image

from train.hyperparameters import DATA_PARAMS
from train.transforms import TransformedDataset, build_eval_transform, build_train_transform

SIZE = DATA_PARAMS["image_size"]

# keep test_transformed_dataset_reads_hugging_face_sample as it is


def test_image_size_is_multiple_of_patch_size():
    assert SIZE % 14 == 0


def test_eval_transform_never_pads_small_images():
    img = Image.new("RGB", (120, 80), (255, 255, 255))  # smaller than the crop
    out = build_eval_transform()(img)
    assert out.shape == (3, SIZE, SIZE)
    # a constant image must stay constant; black padding would break this
    assert torch.allclose(out, out[:, :1, :1].expand_as(out))


def test_eval_transform_is_deterministic():
    img = Image.effect_noise((300, 200), 64).convert("RGB")
    t = build_eval_transform()
    assert torch.equal(t(img), t(img))


def test_train_transform_is_random_but_reproducible_under_seed():
    img = Image.effect_noise((300, 200), 64).convert("RGB")
    t = build_train_transform()
    torch.manual_seed(0); a = t(img)
    torch.manual_seed(0); b = t(img)
    torch.manual_seed(1); c = t(img)
    assert a.shape == (3, SIZE, SIZE)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)