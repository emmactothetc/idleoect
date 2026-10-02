"""Assignment 2 models. Assignment 3 imports these exact implementations."""

from __future__ import annotations

from torch import nn


class MNISTResNet18(nn.Module):
    """ResNet18 adapted for one-channel 28x28 MNIST and ten raw logits."""

    def __init__(self, num_classes=10):
        super().__init__()
        # TODO: construct weights=None ResNet18, replace conv1 for one channel,
        # remove the early max-pool, and replace the classifier.
        raise NotImplementedError("A2: implement MNISTResNet18")

    def forward(self, images):
        raise NotImplementedError("A2: implement MNISTResNet18.forward")


class EuroSATClassifier(nn.Module):
    """Compact 13-band, 10-class CNN returning raw logits."""

    def __init__(self, in_channels=13, num_classes=10, base_channels=32):
        super().__init__()
        # TODO: define a compact classifier and retain the configuration.
        raise NotImplementedError("A2: implement EuroSATClassifier")

    def forward(self, images):
        raise NotImplementedError("A2: implement EuroSATClassifier.forward")


class PlanetUNet(nn.Module):
    """Five-stage U-Net for three-class Planet segmentation.

    Required API: ``PlanetUNet(in_channels=4, num_classes=3, base_channels=8)``.
    Expose ``encoder`` as a five-element ``nn.ModuleList`` and ``bottleneck`` as
    a separate module. Each encoder stage creates a skip before 2x downsampling.
    Implement five decoder stages with corresponding skips. Return three raw
    logits per pixel and preserve the original H,W, including odd input sizes.
    """

    def __init__(self, in_channels=4, num_classes=3, base_channels=8):
        super().__init__()
        # TODO: define five encoder stages, bottleneck, five decoder stages,
        # skip fusion blocks, and the three-channel output projection.
        raise NotImplementedError(
            "A2: implement the required five-stage Planet U-Net"
        )

    def forward(self, images):
        raise NotImplementedError(
            "A2: implement five skips and progressive reconstruction"
        )
