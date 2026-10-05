"""Small, installation-free setup helpers for the progressive assignments."""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

REQUIRED_FILES = (
    "catalog.csv",
    "src/student_pipeline.py",
    "src/student_models.py",
    "src/student_training.py",
    "src/student_inference.py",
    "src/assessment_support.py",
)

DEPENDENCIES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "rasterio": "rasterio",
    "sklearn": "scikit-learn",
    "torch": "torch",
    "torchvision": "torchvision",
    "torchgeo": "torchgeo",
    "gdown": "gdown",
}


@dataclass(frozen=True)
class AssignmentPaths:
    project_root: Path
    data_root: Path
    output_root: Path
    catalog_path: Path


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the complete assignment folder on 2i2c, Colab, or locally."""
    candidates: list[Path] = []
    if start is not None:
        start_path = Path(start).expanduser()
        candidates.extend((start_path, start_path.parent))
    cwd = Path.cwd()
    candidates.extend(
        (
            cwd,
            cwd.parent,
            Path.home() / "assignments",
            Path("/home/jovyan/assignments"),
            Path("/content/assignments"),
            Path("/content/drive/MyDrive/IDLEO/assignments"),
        )
    )
    for candidate in candidates:
        if (candidate / "src" / "student_pipeline.py").is_file() and (
            candidate / "catalog.csv"
        ).is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find the complete assignments folder. On 2i2c "
        "upload/extract "
        "it as /home/jovyan/assignments. On Colab place it in "
        "My Drive/IDLEO/assignments."
    )


def check_dependencies() -> dict[str, str]:
    """Return installed versions or raise with the fallback install command."""
    versions: dict[str, str] = {}
    missing: list[str] = []
    for module_name, package_name in DEPENDENCIES.items():
        try:
            module = importlib.import_module(module_name)
            versions[module_name] = str(
                getattr(module, "__version__", "installed")
            )
        except (ImportError, OSError, RuntimeError) as exc:
            missing.append(f"{package_name} ({exc})")
    if missing:
        raise RuntimeError(
            "Missing or incompatible dependencies: "
            + "; ".join(missing)
            + ". On 2i2c first confirm the selected server image. Then run "
            "Run python -m pip install -r requirements.txt from the "
            "assignments folder. "
            "On Colab run the optional missing-package cell and restart once."
        )
    return versions


def prepare_project(project_root: str | Path | None = None) -> AssignmentPaths:
    """Validate the folder, enable source imports, and prepare outputs."""
    root = find_project_root(project_root)
    missing = [
        relative
        for relative in REQUIRED_FILES
        if not (root / relative).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "The assignment folder is incomplete: " + ", ".join(missing)
        )
    source = str(root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    data_root = (
        Path(os.environ.get("IDLEO_DATA_ROOT", root / "data"))
        .expanduser()
        .resolve()
    )
    output_root = root / "outputs"
    for number in (1, 2, 3):
        (output_root / f"assignment_{number}").mkdir(
            parents=True, exist_ok=True
        )
    return AssignmentPaths(root, data_root, output_root, root / "catalog.csv")
