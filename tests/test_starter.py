"""Checks that should pass before students implement assessed functions."""

from pathlib import Path

import pytest
import torch

import student_pipeline
from assessment_support import ClassCounts
from assignment_setup import find_project_root, prepare_project


def test_folder_is_self_contained():
    root = find_project_root(Path(__file__).parents[1])
    paths = prepare_project(root)
    assert paths.catalog_path.is_file()
    assert paths.project_root == root


def test_assessed_stub_fails_clearly():
    with pytest.raises(NotImplementedError, match="A1"):
        student_pipeline.paired_flip(
            torch.zeros(1, 2, 2), torch.zeros(2, 2, dtype=torch.long), True
        )


def test_multiclass_metric_support_is_provided():
    counts = ClassCounts(3)
    counts.update(torch.tensor([0, 1, 2, 2]), torch.tensor([0, 1, 2, -1]))
    result = counts.compute()
    assert result["support"] == [1, 1, 1]
    assert result["ignored_observations"] == 1


def test_private_material_is_ignored_and_not_required():
    root = Path(__file__).parents[1]
    text = (root / ".gitignore").read_text(encoding="utf-8")
    assert "answers/" in text
    assert "*.pth" in text
