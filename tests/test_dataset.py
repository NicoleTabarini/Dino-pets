import sys
from types import SimpleNamespace

from data.dataset import build_datasets


def test_build_datasets_uses_seeded_stratified_split(monkeypatch):
    train_rows = [{"image": "train-image", "label": 0}]
    validation_rows = [{"image": "validation-image", "label": 1}]

    class FakeTrainSplit:
        features = {"label": SimpleNamespace(names=["cat", "dog"])}

        def train_test_split(self, **kwargs):
            self.kwargs = kwargs
            return {"train": train_rows, "test": validation_rows}

    train_split = FakeTrainSplit()
    test_split = [{"image": "test-image", "label": 1}]
    calls = []

    def fake_load_dataset(name):
        calls.append(name)
        return {"train": train_split, "test": test_split}

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=fake_load_dataset),
    )

    train, validation, test, info = build_datasets(
        dataset_name="timm/oxford-iiit-pet",
        val_frac=0.15,
        seed=42,
    )

    assert calls == ["timm/oxford-iiit-pet"]
    assert train_split.kwargs == {
        "test_size": 0.15,
        "seed": 42,
        "stratify_by_column": "label",
    }
    assert train[0] == {"image": "train-image", "label": 0}
    assert validation[0] == {"image": "validation-image", "label": 1}
    assert test[0] == {"image": "test-image", "label": 1}
    assert info == {
        "num_classes": 2,
        "classes": ["cat", "dog"],
        "n_train": 1,
        "n_val": 1,
        "n_test": 1,
    }

def test_real_split_is_stratified_disjoint_and_deterministic(monkeypatch):
    datasets = pytest.importorskip("datasets")
    labels = [c for c in range(4) for _ in range(10)]
    ds = datasets.Dataset.from_dict(
        {"idx": list(range(40)), "label": labels},
        features=datasets.Features({
            "idx": datasets.Value("int64"),
            "label": datasets.ClassLabel(names=["a", "b", "c", "d"]),
        }),
    )
    monkeypatch.setattr(datasets, "load_dataset", lambda name, revision=None: {"train": ds, "test": ds})

    train, val, _, _ = build_datasets("x", val_frac=0.2, seed=0)
    train2, val2, _, _ = build_datasets("x", val_frac=0.2, seed=0)

    assert set(train["idx"]).isdisjoint(val["idx"])
    assert set(train["label"]) == set(val["label"]) == {0, 1, 2, 3}
    assert val["idx"] == val2["idx"] and train["idx"] == train2["idx"]