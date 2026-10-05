"""Run rubric-mapped behavioral checks against one student submission."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

GRADING_ROOT = Path(__file__).resolve().parent

AUTOMATED = {
    1: (
        (
            "Transforms and normalization",
            10,
            "test_assignment_1.py::test_transforms_and_normalization",
        ),
        (
            "MNIST loaders and exact replay",
            15,
            "test_assignment_1.py::test_mnist_loaders",
        ),
        (
            "EuroSAT100 loaders and official membership",
            12,
            "test_assignment_1.py::test_eurosat_loaders",
        ),
        (
            "Planet catalog and directory datasets",
            20,
            "test_assignment_1.py::test_planet_datasets",
        ),
        (
            "Planet loaders and exact replay",
            13,
            "test_assignment_1.py::test_planet_loaders",
        ),
        (
            "Portable data protocol",
            10,
            "test_assignment_1.py::test_protocol_round_trip",
        ),
    ),
    2: (
        (
            "Exact A1/A2 handoff artifacts",
            10,
            "test_assignment_2.py::test_progressive_handoff_artifacts",
        ),
        (
            "Adapted MNIST ResNet18",
            12,
            "test_assignment_2.py::test_mnist_resnet18",
        ),
        (
            "EuroSAT classifier",
            10,
            "test_assignment_2.py::test_eurosat_classifier",
        ),
        (
            "Five-stage three-class Planet U-Net",
            25,
            "test_assignment_2.py::test_five_stage_unet",
        ),
        (
            "Complete train/evaluation epoch",
            15,
            "test_assignment_2.py::test_run_epoch",
        ),
        (
            "Validation-selected checkpoint and history",
            10,
            "test_assignment_2.py::test_fit",
        ),
    ),
    3: (
        (
            "Protocol, source, and checkpoint provenance",
            15,
            "test_assignment_3.py::test_provenance_artifacts",
        ),
        (
            "Held-out classification inference",
            20,
            "test_assignment_3.py::test_classification_inference",
        ),
        (
            "Georeferenced three-class export",
            10,
            "test_assignment_3.py::test_geotiff_export",
        ),
        (
            "Complete Planet inference/export workflow",
            15,
            "test_assignment_3.py::test_planet_inference_and_export",
        ),
    ),
}

MANUAL = {
    1: (
        ("Protocol/shape/figure evidence and interpretation", 10),
        ("Critical understanding answers", 10),
    ),
    2: (
        (
            "Architecture diagrams, shape tables, histories, interpretation",
            8,
        ),
        ("Critical understanding answers", 10),
    ),
    3: (
        ("Predetermined failure analysis and completed model card", 25),
        ("Critical understanding answers", 15),
    ),
}


def _run_check(submission_root: Path, node_id: str) -> dict:
    test_path, test_name = node_id.split("::", 1)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "-c",
        str(submission_root / "pyproject.toml"),
        f"{GRADING_ROOT / 'tests' / test_path}::{test_name}",
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(submission_root / "src"), str(submission_root))
    )
    environment["IDLEO_SUBMISSION_ROOT"] = str(submission_root)
    completed = subprocess.run(
        command,
        cwd=submission_root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    lines = completed.stdout.strip().splitlines()
    return {
        "passed": completed.returncode == 0,
        "exit_code": completed.returncode,
        "output": "\n".join(lines[-35:]),
    }


def grade(
    assignment: int, submission_root: Path, report_dir: Path | None = None
) -> dict:
    submission_root = submission_root.expanduser().resolve()
    required = submission_root / "src" / "student_pipeline.py"
    if not required.is_file():
        raise FileNotFoundError(
            f"Not an assignments submission: {submission_root}"
        )
    report_dir = (report_dir or submission_root / "grading_reports").resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for label, points, node_id in AUTOMATED[assignment]:
        result = _run_check(submission_root, node_id)
        results.append(
            {
                "label": label,
                "points_available": points,
                "points_awarded": points if result["passed"] else 0,
                "test": node_id,
                **result,
            }
        )
    automated_available = sum(item["points_available"] for item in results)
    automated_awarded = sum(item["points_awarded"] for item in results)
    manual_available = sum(points for _, points in MANUAL[assignment])
    document = {
        "assignment": assignment,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "submission_root": str(submission_root),
        "automated_points_awarded": automated_awarded,
        "automated_points_available": automated_available,
        "manual_points_available": manual_available,
        "total_points_available": automated_available + manual_available,
        "automated_results": results,
        "manual_items": [
            {"label": label, "points_available": points}
            for label, points in MANUAL[assignment]
        ],
    }
    json_path = report_dir / f"assignment_{assignment}_grading.json"
    markdown_path = report_dir / f"assignment_{assignment}_grading.md"
    json_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    lines = [
        f"# Assignment {assignment} grading report",
        "",
        f"Automated subtotal: **{automated_awarded}/{automated_available}**",
        f"Manual points remaining: **{manual_available}**",
        "",
        "## Automated checks",
        "",
        "| Result | Criterion | Points |",
        "|---|---|---:|",
    ]
    for item in results:
        status = "PASS" if item["passed"] else "FAIL"
        lines.append(
            f"| {status} | {item['label']} | "
            f"{item['points_awarded']}/{item['points_available']} |"
        )
    lines.extend(["", "## Manual review", ""])
    for label, points in MANUAL[assignment]:
        lines.append(f"- [ ] {label}: __/{points}")
    failed = [item for item in results if not item["passed"]]
    if failed:
        lines.extend(["", "## Failure details", ""])
        for item in failed:
            lines.extend(
                [
                    f"### {item['label']}",
                    "",
                    "```text",
                    item["output"],
                    "```",
                    "",
                ]
            )
    lines.extend(
        [
            "## Final score",
            "",
            f"Automated {automated_awarded}/{automated_available} + "
            f"manual __/{manual_available} = __/100",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print(markdown_path)
    print(f"Automated subtotal: {automated_awarded}/{automated_available}")
    print(f"Manual review remaining: {manual_available} points")
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assignment", type=int, choices=(1, 2, 3))
    parser.add_argument(
        "--submission-root", type=Path, default=GRADING_ROOT.parent
    )
    parser.add_argument("--report-dir", type=Path)
    arguments = parser.parse_args()
    result = grade(
        arguments.assignment, arguments.submission_root, arguments.report_dir
    )
    return (
        0
        if result["automated_points_awarded"]
        == result["automated_points_available"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
