"""Completion checks for student work and private reference implementations."""

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin

import student_inference as inference
import student_pipeline as pipeline
import student_training as training
from assessment_support import verify_prediction_export
from student_models import EuroSATClassifier, MNISTResNet18, PlanetUNet


def test_transforms_preserve_alignment_and_class_two():
    target = torch.tensor([[0, 1, 2], [-1, 2, 0]])
    image = target.float()[None].repeat(4, 1, 1)
    changed_image, changed_target = pipeline.paired_flip(
        image, target, horizontal=True, vertical=True
    )
    assert torch.equal(changed_image[0], changed_target.float())
    assert 2 in changed_target
    assert torch.isfinite(pipeline.normalize_image(torch.ones(4, 3, 3))).all()


def test_protocol_round_trip(tmp_path):
    path = tmp_path / "protocol.json"
    pipeline.save_data_protocol(
        path,
        seed=42,
        mnist_split_ids={"train": [0], "validate": [1], "test": [60_000]},
        eurosat_split_ids={"train": [0], "validate": [1], "test": [2]},
        planet_selected_rows={"train": [10], "validate": [11], "test": [12]},
        settings={"batch_size": 2},
    )
    loaded = pipeline.load_data_protocol(path)
    assert loaded["mnist_split_ids"]["test"] == [60_000]
    assert loaded["planet_selected_rows"]["test"] == [12]


def test_classifier_and_five_stage_unet_shapes():
    mnist_classifier = MNISTResNet18()
    assert mnist_classifier(torch.zeros(2, 1, 28, 28)).shape == (2, 10)
    classifier = EuroSATClassifier(base_channels=4)
    assert classifier(torch.zeros(2, 13, 64, 64)).shape == (2, 10)
    model = PlanetUNet(base_channels=2)
    assert len(model.encoder) == 5
    assert len(model.decoder) == 5
    assert model(torch.zeros(1, 4, 255, 257)).shape == (1, 3, 255, 257)


def test_training_loop_processes_every_batch():
    model = torch.nn.Linear(2, 10)
    batches = [
        (torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([0, 1])),
        (torch.tensor([[1.0, 1.0]]), torch.tensor([2])),
    ]
    result = training.run_epoch(
        model, batches, torch.device("cpu"), "classification"
    )
    assert result["valid_observations"] == 3


def test_three_class_geotiff_export(tmp_path):
    source, destination = tmp_path / "source.tif", tmp_path / "prediction.tif"
    image = np.ones((4, 9, 7), dtype="float32")
    image[:, 0, 0] = -9999
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        height=9,
        width=7,
        count=4,
        dtype="float32",
        nodata=-9999,
        crs="EPSG:32631",
        transform=from_origin(10, 20, 2, 2),
    ) as dataset:
        dataset.write(image)
    prediction = np.zeros((9, 7), dtype="int64")
    prediction[1, 1], prediction[2, 2] = 1, 2
    valid = np.ones((9, 7), dtype=bool)
    valid[0, 0] = False
    inference.export_prediction(prediction, valid, source, destination)
    verify_prediction_export(source, destination)
