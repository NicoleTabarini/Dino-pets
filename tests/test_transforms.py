import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

import torch
from PIL import Image

from train.hyperparameters import DATA_PARAMS
from train.transforms import TransformedDataset, build_eval_transform, build_train_transform

SIZE = DATA_PARAMS["image_size"]

def test_image_size_is_multiple_of_patch_size():
    """Check that the image size fits whole DINOv2 patches."""
    assert SIZE % 14 == 0


def test_eval_transform_never_pads_small_images():
    """Check that small images become the right size without added padding."""
    img = Image.new("RGB", (120, 80), (255, 255, 255))  # smaller than the crop
    out = build_eval_transform()(img)
    assert out.shape == (3, SIZE, SIZE)
    # a constant image must stay constant; black padding would break this
    assert torch.allclose(out, out[:, :1, :1].expand_as(out))


def test_eval_transform_is_deterministic():
    """Check that evaluation gives the same result each time."""
    img = Image.effect_noise((300, 200), 64).convert("RGB")
    t = build_eval_transform()
    assert torch.equal(t(img), t(img))


def test_train_transform_is_random_but_reproducible_under_seed():
    """Check that a seed repeats training transforms and changes them across seeds."""
    img = Image.effect_noise((300, 200), 64).convert("RGB")
    t = build_train_transform()
    torch.manual_seed(0); a = t(img)
    torch.manual_seed(0); b = t(img)
    torch.manual_seed(1); c = t(img)
    assert a.shape == (3, SIZE, SIZE)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)