"""Export the exact answer-free folder students upload to 2i2c or Drive."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

STUDENT_FILES = (
    "README.md",
    "AI_USE.md",
    ".gitignore",
    "requirements.txt",
    "pyproject.toml",
    "assignment_setup.py",
    "catalog.csv",
    "src/assessment_support.py",
    "src/student_checks.py",
    "src/student_pipeline.py",
    "src/student_models.py",
    "src/student_training.py",
    "src/student_inference.py",
    "assignment1.ipynb",
    "assignment2.ipynb",
    "assignment3.ipynb",
    "tests/test_starter.py",
    "completion_tests/test_progressive_workflow.py",
    "grading/README.md",
    "grading/grade_assignment.py",
    "grading/tests/test_assignment_1.py",
    "grading/tests/test_assignment_2.py",
    "grading/tests/test_assignment_3.py",
)


def export_student_bundle(destination):
    source = Path(__file__).resolve().parent
    destination = Path(destination).resolve()
    archive = destination.with_suffix(".zip")
    if destination.exists() or archive.exists():
        raise FileExistsError(
            "Choose a new review destination; existing exports are preserved"
        )
    files = {relative: source / relative for relative in STUDENT_FILES}
    missing = [
        relative for relative, path in files.items() if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing student files: " + ", ".join(missing))
    forbidden = (
        "answers",
        "data/",
        "outputs/",
        ".venv",
        ".pth",
        ".pt",
        "_completed.ipynb",
    )
    for relative in files:
        normalized = relative.replace("\\", "/").lower()
        if any(token in normalized for token in forbidden):
            raise ValueError(f"Forbidden student export path: {relative}")
    for relative, path in files.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED
    ) as output:
        for relative in sorted(files):
            output.write(destination / relative, f"assignments/{relative}")
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    print(export_student_bundle(arguments.destination))
