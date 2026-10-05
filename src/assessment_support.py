"""Provided assessment infrastructure with no assessed implementations."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch

PLANET_DRIVE_URL = (
    "https://drive.google.com/drive/folders/11sdCgNpLtAy9YZrDCCGsNqglRX-mwV0f"
)
PLANET_CLASS_NAMES = ("noncrop", "field interior", "boundary")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_path(root, relative):
    root = Path(root).expanduser().resolve()
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"Catalog path escapes the data root: {relative}")
    return result


def fetch_planet_pairs(catalog_frame, data_root, drive_url=PLANET_DRIVE_URL):
    """Download catalog-selected real Planet pairs from the course Drive."""
    import gdown

    required = sorted(
        set(catalog_frame["window_b"]) | set(catalog_frame["mask"])
    )
    missing = [
        relative
        for relative in required
        if not safe_path(data_root, relative).is_file()
    ]
    if missing:
        entries = gdown.download_folder(
            url=drive_url, quiet=True, skip_download=True
        )
        by_name: dict[str, list[str]] = {}
        for entry in entries or []:
            name = entry.path.replace("\\", "/").rsplit("/", 1)[-1]
            by_name.setdefault(name, []).append(entry.id)
        for relative in missing:
            identifiers = by_name.get(Path(relative).name, [])
            if len(identifiers) != 1:
                raise RuntimeError(
                    f"Missing or ambiguous Google Drive entry for {relative}. "
                    "Use the manual course-data download if the Drive listing "
                    "is unavailable."
                )
            destination = safe_path(data_root, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_name(destination.name + ".part")
            if not gdown.download(
                id=identifiers[0], output=str(partial), quiet=True
            ):
                raise RuntimeError(f"Download failed: {relative}")
            with rasterio.open(partial) as dataset:
                dataset.read(1, window=((0, 1), (0, 1)))
            partial.replace(destination)
    return {"required_files": len(required), "downloaded_files": len(missing)}


class ClassCounts:
    """Global row=true and column=predicted counts with target -1 ignored."""

    def __init__(self, num_classes):
        self.num_classes = int(num_classes)
        self.matrix = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.int64
        )
        self.ignored = 0

    def update(self, prediction, target):
        prediction = torch.as_tensor(prediction).detach().cpu()
        target = torch.as_tensor(target).detach().cpu()
        if prediction.shape != target.shape:
            raise ValueError("Prediction and target shapes must match")
        valid = target != -1
        if ((target[valid] < 0) | (target[valid] >= self.num_classes)).any():
            raise ValueError("Target contains an unknown valid class")
        if (
            (prediction[valid] < 0) | (prediction[valid] >= self.num_classes)
        ).any():
            raise ValueError("Prediction contains an unknown valid class")
        self.ignored += int((~valid).sum())
        encoded = (
            target[valid].long() * self.num_classes + prediction[valid].long()
        )
        self.matrix += torch.bincount(
            encoded, minlength=self.num_classes**2
        ).reshape(self.matrix.shape)

    def compute(self):
        matrix = self.matrix.numpy()
        if not matrix.sum():
            raise ValueError("No valid observations were evaluated")
        true_positive = matrix.diagonal()
        support = matrix.sum(1)
        predicted = matrix.sum(0)

        def ratios(numerator, denominator):
            return [
                float(a / b) if b else None
                for a, b in zip(numerator, denominator)
            ]

        iou = ratios(true_positive, support + predicted - true_positive)
        defined_iou = [value for value in iou if value is not None]
        return {
            "confusion_matrix": matrix.tolist(),
            "support": support.tolist(),
            "accuracy": float(true_positive.sum() / matrix.sum()),
            "iou": iou,
            "dice": ratios(2 * true_positive, support + predicted),
            "recall": ratios(true_positive, support),
            "precision": ratios(true_positive, predicted),
            "mean_iou": float(np.mean(defined_iou)),
            "valid_observations": int(matrix.sum()),
            "ignored_observations": self.ignored,
        }


def verify_prediction_export(source_path, prediction_path):
    """Reopen and independently verify a three-class prediction."""
    with (
        rasterio.open(source_path) as source,
        rasterio.open(prediction_path) as prediction,
    ):
        if source.crs is None:
            raise ValueError("The source raster has no CRS")
        if (source.crs, source.transform, source.shape) != (
            prediction.crs,
            prediction.transform,
            prediction.shape,
        ):
            raise ValueError("Prediction does not preserve the source grid")
        if (
            prediction.count != 1
            or prediction.dtypes != ("uint8",)
            or prediction.nodata != 255
        ):
            raise ValueError(
                "Prediction must be one uint8 band with nodata=255"
            )
        values = prediction.read(1)
        image_valid = (source.read_masks() > 0).all(0) & np.isfinite(
            source.read()
        ).all(0)
        if not np.isin(values[image_valid], (0, 1, 2)).all():
            raise ValueError("Valid output pixels must be classes 0, 1, or 2")
        if not (values[~image_valid] == 255).all():
            raise ValueError("Image-invalid output pixels must be nodata 255")
    return {
        "prediction_path": str(Path(prediction_path)),
        "sha256": sha256(prediction_path),
    }


def show_eurosat_batch(images, labels, max_items=4):
    count = min(max_items, len(images))
    figure, axes = plt.subplots(1, count, figsize=(3 * count, 3))
    axes = np.atleast_1d(axes)
    for axis, image, label in zip(axes, images[:count], labels[:count]):
        rgb = image[
            [3, 2, 1]
            if image.shape[0] > 3
            else list(range(min(3, image.shape[0])))
        ]
        rgb = rgb.detach().cpu().float().numpy().transpose(1, 2, 0)
        rgb = (rgb - np.nanmin(rgb)) / max(
            float(np.nanmax(rgb) - np.nanmin(rgb)), 1e-6
        )
        axis.imshow(rgb)
        axis.set_title(f"class {int(label)}")
        axis.axis("off")
    figure.tight_layout()
    return figure


def show_planet_triplet(image, target, prediction=None):
    panels = 3 if prediction is not None else 2
    figure, axes = plt.subplots(1, panels, figsize=(4 * panels, 4))
    rgb = image[[2, 1, 0]].detach().cpu().float().numpy().transpose(1, 2, 0)
    rgb = (rgb - np.nanmin(rgb)) / max(
        float(np.nanmax(rgb) - np.nanmin(rgb)), 1e-6
    )
    axes[0].imshow(rgb)
    axes[0].set_title("Planet RGB")
    axes[1].imshow(
        target.detach().cpu(),
        vmin=-1,
        vmax=2,
        cmap="viridis",
        interpolation="nearest",
    )
    axes[1].set_title("Reference")
    if prediction is not None:
        axes[2].imshow(
            prediction.detach().cpu(),
            vmin=0,
            vmax=2,
            cmap="viridis",
            interpolation="nearest",
        )
        axes[2].set_title("Prediction")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    return figure
