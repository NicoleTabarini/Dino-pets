import pytest

pytest.importorskip("torch")

from train.hyperparameters import DATA_PARAMS
from train.transforms import TransformedDataset, build_eval_transform


def test_transformed_dataset_reads_hugging_face_sample():
    class FakeImage:
        def convert(self, mode):
            assert mode == "RGB"
            return "rgb-image"

    class FakeDataset:
        def __getitem__(self, index):
            assert index == 0
            return {"image": FakeImage(), "label": 3}

        def __len__(self):
            return 1

    dataset = TransformedDataset(FakeDataset(), lambda image: f"transformed-{image}")

    assert dataset[0] == ("transformed-rgb-image", 3)


def test_eval_transform_resizes_before_center_crop():
    transform = build_eval_transform()

    expected_image_size = DATA_PARAMS["image_size"]
    expected_resize = round(
        expected_image_size * DATA_PARAMS["eval_resize_scale"]
    )
    assert transform.transforms[0].size == expected_resize
    assert transform.transforms[1].size == expected_image_size
