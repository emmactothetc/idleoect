"""Behavioral A3 grading checks; contains no reference implementation."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin
from torch import nn

import student_inference as inference
from assessment_support import verify_prediction_export

ROOT = Path(os.environ["IDLEO_SUBMISSION_ROOT"])


def _sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def test_provenance_artifacts():
    protocol_path = ROOT / "outputs/assignment_1/data_protocol.json"
    manifest_path = ROOT / "outputs/assignment_2/experiment_manifest.json"
    summary_path = ROOT / "outputs/assignment_3/evaluation_summary.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert manifest["data_protocol_sha256"] == _sha256(protocol_path)
    assert summary["data_protocol_sha256"] == _sha256(protocol_path)
    assert summary["experiment_manifest_sha256"] == _sha256(manifest_path)
    for prefix in ("mnist", "classifier", "segmentation"):
        checkpoint = (
            ROOT / "outputs/assignment_2" / manifest[f"{prefix}_checkpoint"]
        )
        assert checkpoint.is_file()
        assert manifest[f"{prefix}_checkpoint_sha256"] == _sha256(checkpoint)


class _ClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.grad_states = []

    def forward(self, images):
        self.grad_states.append(torch.is_grad_enabled())
        labels = images[:, 0].long()
        logits = torch.full((len(labels), 10), -10.0, device=images.device)
        logits.scatter_(1, labels[:, None], 10.0)
        return logits


def test_classification_inference():
    model = _ClassificationModel()
    loader = [
        (torch.tensor([[0.0], [1.0]]), torch.tensor([0, 1])),
        (torch.tensor([[2.0]]), torch.tensor([2])),
    ]
    result = inference.predict_classification(
        model, loader, torch.device("cpu")
    )
    assert result["predictions"] == [0, 1, 2]
    assert result["targets"] == [0, 1, 2]
    assert result["metrics"]["valid_observations"] == 3
    assert model.training is False and model.grad_states == [False, False]


def _write_source(path, class_map):
    values = np.ones((4, *class_map.shape), dtype="float32")
    values[0] = class_map
    values[:, 0, 0] = -9999
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=class_map.shape[0],
        width=class_map.shape[1],
        count=4,
        dtype="float32",
        nodata=-9999,
        crs="EPSG:32631",
        transform=from_origin(10, 20, 2, 2),
    ) as dataset:
        dataset.write(values)
    return values


def test_geotiff_export(tmp_path):
    source, destination = tmp_path / "source.tif", tmp_path / "prediction.tif"
    prediction = np.zeros((9, 7), dtype="int64")
    prediction[1, 1], prediction[2, 2] = 1, 2
    _write_source(source, prediction)
    valid = np.ones(prediction.shape, dtype=bool)
    valid[0, 0] = False
    result = inference.export_prediction(prediction, valid, source, destination)
    assert Path(result) == destination
    verify_prediction_export(source, destination)
    with rasterio.open(destination) as dataset:
        assert 2 in dataset.read(1)


class _MapModel(nn.Module):
    def forward(self, images):
        classes = images[:, 0].round().clamp(0, 2).long()
        logits = torch.full(
            (len(images), 3, *images.shape[-2:]), -10.0, device=images.device
        )
        return logits.scatter_(1, classes[:, None], 10.0)


def test_planet_inference_and_export(tmp_path):
    maps = []
    sources = []
    for index in range(2):
        class_map = np.zeros((8, 8), dtype="float32")
        class_map[1:4, 1:4], class_map[4:6, 4:6] = 1, 2
        source = tmp_path / f"source_{index}.tif"
        values = _write_source(source, class_map)
        maps.append(
            (torch.from_numpy(values), torch.from_numpy(class_map).long())
        )
        sources.append(str(source))
    images = torch.stack([item[0] for item in maps])
    targets = torch.stack([item[1] for item in maps])
    targets[:, 0, 0] = -1
    result = inference.predict_and_export(
        _MapModel(),
        [(images, targets, sources)],
        torch.device("cpu"),
        tmp_path / "predictions",
    )
    assert result["metrics"]["valid_observations"] == 2 * (8 * 8 - 1)
    assert result["metrics"]["support"][2] > 0 and len(result["files"]) == 2
    for record in result["files"]:
        path = Path(record["prediction_path"])
        assert path.is_file() and record["sha256"] == _sha256(path)
