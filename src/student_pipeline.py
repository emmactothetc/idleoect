"""Assignment 1 implementations, reused unchanged by Assignments 2 and 3."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def paired_flip(
    image: torch.Tensor,
    target: torch.Tensor,
    horizontal: bool = False,
    vertical: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the same requested flips to a [C,H,W] image and [H,W] target."""
    if image.ndim != 3:
        raise ValueError(f"Expected image [C,H,W], got {image.shape}")
    if target.ndim != 2:
        raise ValueError(f"Expected target [H,W], got {target.shape}")
    if image.shape[1:] != target.shape:
        raise ValueError(
            f"Image spatial {image.shape[1:]} != target {target.shape}"
        )

    image = image.clone()
    target = target.clone()

    dims = []
    if horizontal:
        dims.append(-1)
    if vertical:
        dims.append(-2)

    if dims:
        image = torch.flip(image, dims)
        target = torch.flip(target, dims)

    return image, target


def normalize_image(
    image: torch.Tensor, valid: torch.Tensor | None = None
) -> torch.Tensor:
    """Return per-band float32 z-scores computed over valid image pixels only.

    Zero-fill invalid image pixels. Protect constant and all-invalid bands from
    NaN/Inf. Label availability must never influence image normalization.
    """
    if image.ndim != 3:
        raise ValueError(f"Expected image [C,H,W], got {image.shape}")

    C, H, W = image.shape
    image = image.clone().to(torch.float32)

    if valid is None:
        valid = torch.isfinite(image).all(dim=0)

    if valid.ndim != 2 or valid.shape != (H, W):
        raise ValueError(f"Valid mask must be [H,W]={H},{W}, got {valid.shape}")

    normalized = torch.zeros_like(image)

    for c in range(C):
        band = image[c]
        band_valid = band[valid]

        if band_valid.numel() == 0:
            normalized[c].fill_(0)
            continue

        mean = band_valid.mean()
        std = band_valid.std()

        if std == 0 or not torch.isfinite(std):
            normalized[c][valid] = 0
            normalized[c][~valid] = 0
        else:
            normalized[c] = (band - mean) / std
            normalized[c][~valid] = 0

    return normalized


def training_transform(
    image: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply paired stochastic augmentation after preprocessing."""
    horizontal = random.random() < 0.5
    vertical = random.random() < 0.5
    return paired_flip(image, target, horizontal, vertical)


def evaluation_transform(
    image: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic copies for validation and test data."""
    return image.clone(), target.clone()


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def build_mnist_loaders(
    root: str | Path,
    batch_size: int = 64,
    seed: int = 42,
    split_ids: dict[str, list[int]] | None = None,
    limits: dict[str, int] | None = None,
    download: bool = True,
) -> dict[str, Any]:
    """Build reusable MNIST train/validate/test loaders."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )

    full_train = datasets.MNIST(
        root=root, train=True, download=download, transform=train_transform
    )
    test_dataset = datasets.MNIST(
        root=root, train=False, download=download, transform=eval_transform
    )

    if split_ids is None:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(full_train), generator=generator).tolist()
        split_idx = int(0.9 * len(full_train))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        split_ids = {
            "train": train_indices,
            "validate": val_indices,
            "test": list(range(60000, 60000 + len(test_dataset))),
        }
    else:
        train_indices = split_ids["train"]
        val_indices = split_ids["validate"]

    if limits:
        train_indices = train_indices[: limits.get("train", len(train_indices))]
        val_indices = val_indices[: limits.get("validate", len(val_indices))]
        test_limit = limits.get("test", len(test_dataset))
        test_indices = list(range(test_limit))
    else:
        test_indices = list(range(len(test_dataset)))

    train_subset = Subset(full_train, train_indices)
    val_subset = Subset(full_train, val_indices)
    test_subset = Subset(test_dataset, test_indices)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return {
        "train": train_loader,
        "validate": val_loader,
        "test": test_loader,
        "split_ids": split_ids,
    }


def build_eurosat_loaders(
    root: str | Path,
    batch_size: int = 16,
    seed: int = 42,
    split_ids: dict[str, list[int]] | None = None,
    limits: dict[str, int] | None = None,
    download: bool = True,
) -> dict[str, Any]:
    """Build reproducible EuroSAT100 train/validate/test loaders."""
    from torchgeo.datasets import EuroSAT100

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    if split_ids is None:
        split_ids = {
            "train": list(range(60)),
            "validate": list(range(60, 80)),
            "test": list(range(80, 100)),
        }

    train_dataset = EuroSAT100(root=root, split="train", download=download)
    val_dataset = EuroSAT100(root=root, split="val", download=download)
    test_dataset = EuroSAT100(root=root, split="test", download=download)

    class _DictToTuple(Dataset):
        def __init__(self, ds):
            self.ds = ds

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            sample = self.ds[idx]
            if isinstance(sample, dict):
                return sample["image"], sample["label"]
            return sample

    train_dataset = _DictToTuple(train_dataset)
    val_dataset = _DictToTuple(val_dataset)
    test_dataset = _DictToTuple(test_dataset)

    # Compute normalization stats on training set (before limits)
    train_images = torch.stack(
        [train_dataset[i][0] for i in range(len(train_dataset))]
    )
    mean = train_images.mean(dim=(0, 2, 3))
    std = train_images.std(dim=(0, 2, 3))
    std[std == 0] = 1

    class _Normalize(Dataset):
        def __init__(self, ds, mean, std):
            self.ds = ds
            self.mean = mean
            self.std = std

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            image, label = self.ds[idx]
            image = (image - self.mean[:, None, None]) / self.std[:, None, None]
            return image, label

    train_dataset = _Normalize(train_dataset, mean, std)
    val_dataset = _Normalize(val_dataset, mean, std)
    test_dataset = _Normalize(test_dataset, mean, std)

    if limits:
        train_dataset = Subset(
            train_dataset,
            range(min(limits.get("train", 60), len(train_dataset))),
        )
        val_dataset = Subset(
            val_dataset,
            range(min(limits.get("validate", 20), len(val_dataset))),
        )
        test_dataset = Subset(
            test_dataset, range(min(limits.get("test", 20), len(test_dataset)))
        )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return {
        "train": train_loader,
        "validate": val_loader,
        "test": test_loader,
        "split_ids": split_ids,
    }


class PlanetCatalogDataset(Dataset):
    """Lazy three-class Planet image/mask dataset based on catalog pairs."""

    def __init__(
        self,
        catalog_path: str | Path,
        data_root: str | Path,
        split: str,
        transform=None,
        selected_rows: list[int] | None = None,
        max_items: int | None = None,
    ):
        self.catalog_path = Path(catalog_path)
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.max_items = max_items

        catalog = pd.read_csv(self.catalog_path)
        split_mask = catalog["split"] == split

        if selected_rows is not None:
            # selected_rows are global catalog indices; intersect with split
            selected_set = set(selected_rows)
            split_rows = catalog[split_mask & catalog.index.isin(selected_set)]
        else:
            split_rows = catalog[split_mask]

        if max_items is not None:
            split_rows = split_rows.head(max_items)

        self.rows = split_rows.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.rows.iloc[index]
        image_path = self.data_root / row["window_b"]
        mask_path = self.data_root / row["mask"]

        with rasterio.open(image_path) as img_src:
            image = img_src.read().astype(np.float32)
            img_crs = img_src.crs
            img_transform = img_src.transform
            img_shape = img_src.shape

        with rasterio.open(mask_path) as msk_src:
            mask = msk_src.read(1)
            msk_crs = msk_src.crs
            msk_transform = msk_src.transform
            msk_shape = msk_src.shape

        if img_crs != msk_crs:
            raise ValueError(f"CRS mismatch: {img_crs} != {msk_crs}")
        if img_transform != msk_transform:
            raise ValueError("Transform mismatch")
        if img_shape != msk_shape:
            raise ValueError(f"Shape mismatch: {img_shape} != {msk_shape}")

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask).to(torch.int64)

        valid = torch.isfinite(image).all(dim=0)
        mask[~valid] = -1

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask, str(image_path.absolute())


def match_directory_pairs(
    image_dir: str | Path,
    label_dir: str | Path,
    allowed_image_names: list[str],
) -> list[tuple[Path, Path]]:
    """Independently match permitted image/mask filenames by site and date."""
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)

    image_files = {f.stem: f for f in image_dir.glob("*.tif")}
    label_files = {f.stem: f for f in label_dir.glob("*.tif")}

    pairs = []
    for name in allowed_image_names:
        name_stem = Path(name).stem
        if name_stem not in image_files:
            continue

        parts = name_stem.split("_")
        if len(parts) < 2:
            continue
        site = parts[0]
        date = "_".join(parts[1:])

        matching_labels = [
            (stem, path)
            for stem, path in label_files.items()
            if stem.startswith(site + "_") and stem.endswith("_" + date)
        ]

        if len(matching_labels) != 1:
            continue

        pairs.append((image_files[name_stem], matching_labels[0][1]))

    return pairs


class PlanetDirectoryDataset(Dataset):
    """Lazy directory-pair dataset with the catalog-loading item contract."""

    def __init__(self, pairs: list[tuple[Path, Path]], transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        image_path, mask_path = self.pairs[index]

        with rasterio.open(image_path) as img_src:
            image = img_src.read().astype(np.float32)
            img_crs = img_src.crs
            img_transform = img_src.transform
            img_shape = img_src.shape

        with rasterio.open(mask_path) as msk_src:
            mask = msk_src.read(1)
            msk_crs = msk_src.crs
            msk_transform = msk_src.transform
            msk_shape = msk_src.shape

        if img_crs != msk_crs:
            raise ValueError(f"CRS mismatch: {img_crs} != {msk_crs}")
        if img_transform != msk_transform:
            raise ValueError("Transform mismatch")
        if img_shape != msk_shape:
            raise ValueError(f"Shape mismatch: {img_shape} != {msk_shape}")

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask).to(torch.int64)

        valid = torch.isfinite(image).all(dim=0)
        mask[~valid] = -1

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask, str(image_path.absolute())


def build_planet_loaders(
    catalog_path: str | Path,
    data_root: str | Path,
    batch_size: int = 2,
    seed: int = 42,
    selected_rows: dict[str, list[int]] | None = None,
    limits: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build official-split Planet loaders and return exact row membership."""
    catalog_path = Path(catalog_path)
    data_root = Path(data_root)

    catalog = pd.read_csv(catalog_path)

    if selected_rows is None:
        train_indices = catalog[catalog["split"] == "train"].index.tolist()
        val_indices = catalog[catalog["split"] == "validate"].index.tolist()
        test_indices = catalog[catalog["split"] == "test"].index.tolist()

        if limits:
            train_indices = train_indices[
                : limits.get("train", len(train_indices))
            ]
            val_indices = val_indices[
                : limits.get("validate", len(val_indices))
            ]
            test_indices = test_indices[: limits.get("test", len(test_indices))]

        selected_rows = {
            "train": train_indices,
            "validate": val_indices,
            "test": test_indices,
        }
    else:
        train_indices = selected_rows["train"]
        val_indices = selected_rows["validate"]
        test_indices = selected_rows["test"]

    train_dataset = PlanetCatalogDataset(
        catalog_path, data_root, "train", training_transform, train_indices
    )
    val_dataset = PlanetCatalogDataset(
        catalog_path, data_root, "validate", evaluation_transform, val_indices
    )
    test_dataset = PlanetCatalogDataset(
        catalog_path, data_root, "test", evaluation_transform, test_indices
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return {
        "train": train_loader,
        "validate": val_loader,
        "test": test_loader,
        "selected_rows": selected_rows,
    }


def save_data_protocol(
    path: str | Path,
    *,
    seed: int,
    eurosat_split_ids: dict[str, list[int]],
    planet_selected_rows: dict[str, list[int]],
    settings: dict[str, Any],
    mnist_split_ids: dict[str, list[int]] | None = None,
) -> None:
    """Save portable membership for exact A2/A3 reconstruction."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    protocol = {
        "seed": seed,
        "eurosat_split_ids": eurosat_split_ids,
        "planet_selected_rows": planet_selected_rows,
        "settings": settings,
        "planet_classes": {
            "0": "noncrop",
            "1": "field interior",
            "2": "boundary",
        },
    }

    if mnist_split_ids is not None:
        protocol["mnist_split_ids"] = mnist_split_ids

    with open(path, "w") as f:
        json.dump(protocol, f, indent=2)


def load_data_protocol(path: str | Path) -> dict[str, Any]:
    """Load and validate the A1 protocol without silently inventing defaults."""
    path = Path(path)
    with open(path) as f:
        protocol = json.load(f)

    required = [
        "seed",
        "eurosat_split_ids",
        "planet_selected_rows",
        "settings",
        "planet_classes",
    ]
    for key in required:
        if key not in protocol:
            raise ValueError(f"Missing required key in protocol: {key}")

    return protocol
