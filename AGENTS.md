# IDLEO 2026 Progressive Assignments — Agent Guidance

## Repository Structure
- **Notebooks**: `assignment1.ipynb`, `assignment2.ipynb`, `assignment3.ipynb` — orchestrate workflow, contain rubrics and evidence cells
- **Student implementations** (edit these):
  - A1: `src/student_pipeline.py` — data pipelines, transforms, loaders, protocol
  - A2: `src/student_models.py` — MNISTResNet18, EuroSATClassifier, PlanetUNet
  - A2: `src/student_training.py` — `run_epoch`, `fit`
  - A3: `src/student_inference.py` — `predict_classification`, `export_prediction`, `predict_and_export`
- **Support (do not edit)**: `src/assessment_support.py`, `src/student_checks.py`, `src/assignment_setup.py`
- **Tests**: `tests/test_starter.py` (pass before implementation), `completion_tests/test_progressive_workflow.py` (pass after)
- **Grading**: `grading/grade_assignment.py {1,2,3} --submission-root /path/to/assignments`

## Key Commands
```bash
# Run starter tests (should pass before implementing)
PYTHONPATH=src python -m pytest tests

# Run completion tests (expected to fail until done)
PYTHONPATH=src python -m pytest completion_tests

# Lint / format
python -m ruff check src tests
python -m ruff format src tests

# Grade a submission (from instructor copy)
python grading/grade_assignment.py 1 --submission-root /path/to/student/assignments
```

## Progressive Workflow Constraints
- **A2 imports A1's module** — must reconstruct exact saved membership from `data_protocol.json`, do not copy/rewrite dataloader code
- **A3 imports A1 + A2 modules** — verifies saved hashes, reconstructs exact model configs, loads validation-selected checkpoints
- **Handoff artifacts** (required for grading):
  - A1: `data_protocol.json` (seed, split IDs, settings)
  - A2: `experiment_manifest.json` (hashes of A1 protocol, shared source, checkpoints), training histories, checkpoints
- **Never** edit `assignment_setup.py`, `student_checks.py`, `assessment_support.py`, the catalog, or tests

## Testing Quirks
- Starter tests verify environment and stub failures; completion tests verify full behavior
- Run from repo root with `PYTHONPATH=src` (configured in `pyproject.toml`)
- Completion tests use `tmp_path` fixtures — safe to run repeatedly
- Grading runs pytest against submission with isolated PYTHONPATH

## Environment
- Target: Python ≥3.10 on 2i2c JupyterHub (path `/home/jovyan/assignments`)
- Dependencies in `requirements.txt` / `pyproject.toml`: numpy, pandas, matplotlib, rasterio, torch, torchvision, torchgeo, scikit-learn, pytest, gdown, nbformat, nbclient, ipywidgets
- No editable install, venv, or custom kernel needed
- Colab fallback: place at `My Drive/IDLEO/assignments`, mount Drive, install only missing packages

## Agent Usage Rules (from README)
- Agents may help with planning, one named TODO, debugging a failing check, or reviewing code
- Do not ask an agent to complete the entire assignment
- Inspect every suggested change, add an independent counterexample, explain the final implementation yourself
- Record prompts, accepted/rejected suggestions, changes, and verification in `AI_USE.md`