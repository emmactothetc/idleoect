"""Behavioral A2 grading checks; contains no reference implementation."""

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import torch
from torch import nn

import student_training as training
from student_models import EuroSATClassifier, MNISTResNet18, PlanetUNet

ROOT = Path(os.environ["IDLEO_SUBMISSION_ROOT"])


def _sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def test_progressive_handoff_artifacts():
    protocol_path = ROOT / "outputs/assignment_1/data_protocol.json"
    manifest_path = ROOT / "outputs/assignment_2/experiment_manifest.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(protocol) >= {
        "mnist_split_ids",
        "eurosat_split_ids",
        "planet_selected_rows",
        "settings",
    }
    assert manifest["data_protocol_sha256"] == _sha256(protocol_path)
    for name in ("student_pipeline", "student_models", "student_training"):
        assert manifest[f"{name}_sha256"] == _sha256(ROOT / f"src/{name}.py")
    for history in (
        "mnist_history.csv",
        "eurosat_history.csv",
        "planet_history.csv",
    ):
        frame = pd.read_csv(ROOT / "outputs/assignment_2" / history)
        assert (
            set(frame) >= {"epoch", "train_loss", "validation_loss"}
            and len(frame) >= 1
        )


def test_mnist_resnet18():
    model = MNISTResNet18()
    result = model(torch.randn(2, 1, 28, 28))
    assert result.shape == (2, 10)
    assert model.config == {"num_classes": 10}
    assert model.model.conv1.in_channels == 1
    assert model.model.conv1.kernel_size == (
        3,
        3,
    ) and model.model.conv1.stride == (1, 1)
    assert isinstance(model.model.maxpool, nn.Identity)
    assert model.model.fc.out_features == 10


def test_eurosat_classifier():
    model = EuroSATClassifier(in_channels=13, num_classes=10, base_channels=4)
    assert model(torch.randn(3, 13, 64, 64)).shape == (3, 10)
    assert model.config == {
        "in_channels": 13,
        "num_classes": 10,
        "base_channels": 4,
    }
    first_convolution = next(
        layer for layer in model.modules() if isinstance(layer, nn.Conv2d)
    )
    assert first_convolution.in_channels == 13


def test_five_stage_unet():
    model = PlanetUNet(in_channels=4, num_classes=3, base_channels=2)
    assert isinstance(model.encoder, nn.ModuleList) and len(model.encoder) == 5
    assert isinstance(model.bottleneck, nn.Module)
    assert len(model.decoder) == 5
    assert model.config == {
        "in_channels": 4,
        "num_classes": 3,
        "base_channels": 2,
    }
    for height, width in ((256, 256), (255, 257)):
        result = model(torch.randn(1, 4, height, width))
        assert result.shape == (1, 3, height, width)
        result.mean().backward()
        model.zero_grad(set_to_none=True)


def test_run_epoch():
    model = nn.Linear(2, 10)
    batches = [
        (torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([0, 1])),
        (torch.tensor([[1.0, 1.0]]), torch.tensor([2])),
    ]
    before = {name: value.clone() for name, value in model.state_dict().items()}
    result = training.run_epoch(
        model, batches, torch.device("cpu"), "classification"
    )
    assert result["valid_observations"] == 3
    assert all(
        torch.equal(before[name], value)
        for name, value in model.state_dict().items()
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    training.run_epoch(
        model, batches, torch.device("cpu"), "classification", optimizer
    )
    assert any(
        not torch.equal(before[name], value)
        for name, value in model.state_dict().items()
    )
    segmenter = nn.Conv2d(1, 3, 1)
    segmentation = [
        (
            torch.randn(2, 1, 4, 4),
            torch.tensor(
                [
                    [[0, 1, 2, -1]] * 4,
                    [[2, 1, 0, -1]] * 4,
                ]
            ),
        )
    ]
    metrics = training.run_epoch(
        segmenter, segmentation, torch.device("cpu"), "segmentation"
    )
    assert metrics["valid_observations"] == 24 and len(metrics["iou"]) == 3


def test_fit(tmp_path):
    model = nn.Linear(2, 10)
    batches = [(torch.randn(4, 2), torch.tensor([0, 1, 2, 3]))]
    checkpoint = tmp_path / "nested" / "model.pth"
    history = training.fit(
        model,
        batches,
        batches,
        torch.device("cpu"),
        "classification",
        torch.optim.SGD(model.parameters(), lr=0.01),
        2,
        checkpoint,
        {"in_features": 2, "num_classes": 10},
    )
    assert isinstance(history, pd.DataFrame) and len(history) == 2
    assert set(history) >= {"epoch", "train_loss", "validation_loss"}
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert set(saved) >= {
        "model_state",
        "model_config",
        "epoch",
        "task",
        "validation_loss",
    }
    assert saved["model_config"] == {"in_features": 2, "num_classes": 10}
    assert saved["task"] == "classification"
