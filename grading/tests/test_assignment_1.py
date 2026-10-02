"""Behavioral A1 grading checks; contains no reference implementation."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import rasterio
import torch
from PIL import Image
from rasterio.transform import from_origin
from torch.utils.data import Dataset

import student_pipeline as pipeline


def test_transforms_and_normalization():
    target = torch.tensor([[0, 1, 2], [-1, 2, 0]])
    image = target.float()[None].repeat(4, 1, 1)
    original = image.clone()
    changed_image, changed_target = pipeline.paired_flip(
        image, target, True, True
    )
    assert torch.equal(changed_target, torch.flip(target, (-1, -2)))
    assert torch.equal(changed_image[0], changed_target.float())
    assert torch.equal(image, original), (
        "Transforms must not modify caller-owned input"
    )
    valid = torch.tensor([[True, True], [False, False]])
    values = torch.tensor([[[1.0, 3.0], [1000.0, float("nan")]]]).repeat(
        4, 1, 1
    )
    normalized = pipeline.normalize_image(values, valid)
    assert (
        normalized.dtype == torch.float32 and torch.isfinite(normalized).all()
    )
    assert normalized[:, 1].eq(0).all()
    first, second = pipeline.evaluation_transform(image, target)
    assert torch.equal(first, image) and torch.equal(second, target)


class _FakeMNIST(Dataset):
    def __init__(self, root, train, transform, download):
        self.train, self.transform = train, transform
        self.length = 20 if train else 5

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        array = np.full((28, 28), index % 255, dtype=np.uint8)
        image = Image.fromarray(array, mode="L")
        return self.transform(image), index % 10


def test_mnist_loaders(tmp_path):
    with patch("torchvision.datasets.MNIST", _FakeMNIST):
        result = pipeline.build_mnist_loaders(
            tmp_path, batch_size=4, seed=7, download=False
        )
        assert len(result["split_ids"]["train"]) == 18
        assert len(result["split_ids"]["validate"]) == 2
        assert result["split_ids"]["test"] == list(range(60_000, 60_005))
        replay = pipeline.build_mnist_loaders(
            tmp_path,
            batch_size=4,
            seed=999,
            split_ids=result["split_ids"],
            download=False,
        )
        assert replay["split_ids"] == result["split_ids"]
        images, labels = next(iter(result["validate"]))
        assert images.shape[1:] == (1, 28, 28) and labels.dtype == torch.int64
        assert (
            result["train"].num_workers == 0 and result["test"].num_workers == 0
        )


class _FakeEuroSAT100(Dataset):
    sizes = {"train": 60, "val": 20, "test": 20}

    def __init__(self, root, split, download):
        self.split = split

    def __len__(self):
        return self.sizes[self.split]

    def __getitem__(self, index):
        return {"image": torch.full((13, 8, 8), 1000.0), "label": index % 10}


def test_eurosat_loaders(tmp_path):
    with patch("torchgeo.datasets.EuroSAT100", _FakeEuroSAT100):
        result = pipeline.build_eurosat_loaders(
            tmp_path, batch_size=5, seed=3, download=False
        )
        assert result["split_ids"] == {
            "train": list(range(60)),
            "validate": list(range(60, 80)),
            "test": list(range(80, 100)),
        }
        replay = pipeline.build_eurosat_loaders(
            tmp_path,
            batch_size=5,
            seed=999,
            split_ids=result["split_ids"],
            download=False,
        )
        assert replay["split_ids"] == result["split_ids"]
        images, labels = next(iter(result["test"]))
        assert images.shape[1:] == (13, 8, 8) and labels.dtype == torch.int64
        assert float(images.max()) <= 1.0


def _write_pair(root: Path, name: str, split: str, index: int):
    image_dir, label_dir = root / "images", root / "labels"
    image_dir.mkdir(exist_ok=True)
    label_dir.mkdir(exist_ok=True)
    image_path = image_dir / f"{name}_2022-0{index + 1}.tif"
    label_path = label_dir / f"{name}_{100 + index}_2022-0{index + 1}.tif"
    transform = from_origin(10, 20, 2, 2)
    image = np.arange(4 * 8 * 8, dtype="float32").reshape(4, 8, 8) + index
    target = np.zeros((8, 8), dtype="uint8")
    target[1:4, 1:4], target[4:6, 4:6] = 1, 2
    with rasterio.open(
        image_path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=4,
        dtype="float32",
        crs="EPSG:32631",
        transform=transform,
    ) as destination:
        destination.write(image)
    with rasterio.open(
        label_path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=1,
        dtype="uint8",
        crs="EPSG:32631",
        transform=transform,
        nodata=255,
    ) as destination:
        destination.write(target, 1)
    return {
        "name": name,
        "window_b": str(image_path.relative_to(root)).replace("\\", "/"),
        "mask": str(label_path.relative_to(root)).replace("\\", "/"),
        "split": split,
    }


def _planet_fixture(tmp_path):
    rows = [
        _write_pair(tmp_path, f"AO{i:07d}", split, i)
        for i, split in enumerate(
            ("train", "train", "validate", "validate", "test", "test")
        )
    ]
    catalog = tmp_path / "catalog.csv"
    pd.DataFrame(rows).to_csv(catalog, index=False)
    return catalog, rows


def test_planet_datasets(tmp_path):
    catalog, rows = _planet_fixture(tmp_path)
    dataset = pipeline.PlanetCatalogDataset(catalog, tmp_path, "train")
    image, target, source = dataset[0]
    assert image.shape == (4, 8, 8) and image.dtype == torch.float32
    assert target.shape == (8, 8) and target.dtype == torch.int64
    assert set(torch.unique(target).tolist()) == {0, 1, 2}
    assert Path(source).is_absolute() and torch.isfinite(image).all()
    pairs = pipeline.match_directory_pairs(
        tmp_path / "images",
        tmp_path / "labels",
        [Path(rows[0]["window_b"]).name],
    )
    directory_dataset = pipeline.PlanetDirectoryDataset(pairs)
    second_image, second_target, _ = directory_dataset[0]
    assert second_image.shape == image.shape and torch.equal(
        second_target, target
    )


def test_planet_loaders(tmp_path):
    catalog, _ = _planet_fixture(tmp_path)
    membership = {"train": [1], "validate": [2], "test": [5]}
    result = pipeline.build_planet_loaders(
        catalog, tmp_path, batch_size=1, seed=11, selected_rows=membership
    )
    assert result["selected_rows"] == membership
    for split in ("train", "validate", "test"):
        images, targets, paths = next(iter(result[split]))
        assert images.shape == (1, 4, 8, 8)
        assert targets.shape == (1, 8, 8) and 2 in targets
        assert Path(paths[0]).is_absolute()


def test_protocol_round_trip(tmp_path):
    path = tmp_path / "nested" / "protocol.json"
    expected = {
        "mnist_split_ids": {"train": [0], "validate": [1], "test": [60_000]},
        "eurosat_split_ids": {"train": [0], "validate": [60], "test": [80]},
        "planet_selected_rows": {"train": [1], "validate": [2], "test": [3]},
    }
    pipeline.save_data_protocol(
        path, seed=42, settings={"batch_size": 2}, **expected
    )
    loaded = pipeline.load_data_protocol(path)
    for key, value in expected.items():
        assert loaded[key] == value
    assert loaded["seed"] == 42 and loaded["settings"] == {"batch_size": 2}
    assert loaded["planet_classes"] == {
        "0": "noncrop",
        "1": "field interior",
        "2": "boundary",
    }
