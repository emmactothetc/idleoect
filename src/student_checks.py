"""Short verification helpers used beside tasks; they are not solutions."""

import tempfile
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin

from assessment_support import verify_prediction_export


def check_transforms(pipeline):
    target = torch.tensor([[0, 1, 2], [-1, 2, 0]])
    image = target.float().unsqueeze(0).repeat(4, 1, 1)
    for horizontal, vertical in ((True, False), (False, True), (True, True)):
        dimensions = ([-1] if horizontal else []) + ([-2] if vertical else [])
        transformed_image, transformed_target = pipeline.paired_flip(
            image, target, horizontal, vertical
        )
        assert torch.equal(transformed_target, torch.flip(target, dimensions))
        assert torch.equal(transformed_image[0], transformed_target.float())
    valid = torch.tensor([[True, True], [False, False]])
    values = torch.tensor([[[1.0, 3.0], [1000.0, float("nan")]]])
    normalized = pipeline.normalize_image(values, valid)
    assert torch.isfinite(normalized).all()
    assert normalized[0, 1].eq(0).all()
    assert abs(float(normalized[0, 0].mean())) < 1e-6
    assert torch.isfinite(pipeline.normalize_image(torch.ones(4, 2, 2))).all()


def check_loaders(eurosat, planet, mnist=None):
    assert set(eurosat) >= {"train", "validate", "test", "split_ids"}
    assert set(planet) >= {"train", "validate", "test", "selected_rows"}
    items = [(eurosat, "split_ids"), (planet, "selected_rows")]
    if mnist is not None:
        items.append((mnist, "split_ids"))
    for result, membership in items:
        groups = result[membership]
        assert set(groups) == {"train", "validate", "test"}
        assert not (set(groups["train"]) & set(groups["validate"]))
        assert not (set(groups["train"]) & set(groups["test"]))
        assert not (set(groups["validate"]) & set(groups["test"]))


def check_unet_architecture(model):
    assert (
        isinstance(model.encoder, torch.nn.ModuleList)
        and len(model.encoder) == 5
    )
    assert isinstance(model.bottleneck, torch.nn.Module)
    assert hasattr(model, "decoder") and len(model.decoder) == 5
    for height, width in ((256, 256), (255, 257)):
        result = model(torch.zeros(1, 4, height, width))
        assert result.shape == (1, 3, height, width)


def check_training_loop(training, device):
    model = torch.nn.Linear(2, 10).to(device)
    batches = [
        (
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
            torch.tensor([0, 1, 2]),
        ),
        (torch.tensor([[2.0, 1.0]]), torch.tensor([3])),
    ]
    before = {key: value.clone() for key, value in model.state_dict().items()}
    result = training.run_epoch(model, batches, device, "classification")
    assert result["valid_observations"] == 4
    assert all(
        torch.equal(before[key], value)
        for key, value in model.state_dict().items()
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    training.run_epoch(model, batches, device, "classification", optimizer)
    assert any(
        not torch.equal(before[key], value)
        for key, value in model.state_dict().items()
    )


def check_export(inference, scratch_root):
    Path(scratch_root).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="export-check-", dir=scratch_root
    ) as folder:
        source = Path(folder) / "image.tif"
        destination = Path(folder) / "prediction.tif"
        values = np.ones((4, 8, 8), dtype="float32")
        values[:, 0, 0] = -9999
        with rasterio.open(
            source,
            "w",
            driver="GTiff",
            height=8,
            width=8,
            count=4,
            dtype="float32",
            nodata=-9999,
            crs="EPSG:32631",
            transform=from_origin(10, 20, 2, 2),
        ) as dataset:
            dataset.write(values)
        valid = np.ones((8, 8), dtype=bool)
        valid[0, 0] = False
        prediction = np.zeros((8, 8), dtype="int64")
        prediction[1, 1] = 2
        inference.export_prediction(prediction, valid, source, destination)
        verify_prediction_export(source, destination)
