# IDLEO 2026 progressive assignments

Use this complete `assignments` folder for all three assignments. Do not
download or upload individual notebooks: the notebooks import shared source files,
reuse saved split membership, and pass checkpoints forward.

## Start on 2i2c

1. Upload and extract the folder so its path is:

   ```text
   /home/jovyan/assignments
   ```

2. Open `assignment1.ipynb` in JupyterLab.
3. Run the setup cell. It adds `/home/jovyan/assignments/src` to Python's
   import path and checks the environment already supplied by the hub.
4. If and only if the check reports missing packages, open a terminal in the
   assignment folder and run:

   ```bash
   python -m pip install -r requirements.txt
   ```

No editable install, virtual environment, or custom kernel is required. The
`/home/jovyan` directory is persistent on standard 2i2c community hubs, but it is
not a backup. Keep a separate copy of your work.

## Colab fallback

Place the extracted folder at `My Drive/IDLEO/assignments`, open the notebook
in Colab, and run its Drive-mount setup. Keep edited `.py` files, notebooks,
protocols, histories, and checkpoints in Drive. Colab's runtime and temporary
downloads can disappear after a reset. Use its existing PyTorch stack; install
only packages reported missing by the setup check, then restart once.

## One progressive workflow

### Assignment 1: reusable data pipeline

Complete `src/student_pipeline.py`:

- MNIST's packaged train/test pools with a saved seeded 90/10 development split;
- EuroSAT100's official 60/20/20 train/validate/test loaders with seeded training order;
- lazy Planet catalog and directory datasets;
- paired image/mask augmentation and image-only normalization;
- exact three-class masks: `0 noncrop`, `1 field interior`, `2 boundary`;
- portable `data_protocol.json` containing exact membership and settings.

### Assignment 2: models and training

Import A1's module and reconstruct the exact saved membership. Do not copy or
rewrite dataloader code. Complete:

- `EuroSATClassifier` for 13-band, 10-class classification;
- `MNISTResNet18`, adapted for one-channel 28×28 images and reused from A1;
- `PlanetUNet` with five encoder stages, a separate bottleneck, five decoder
  stages, five matching skip connections, and three output logits per pixel;
- complete training/validation loops and minimum-validation-loss checkpoints.

The A2 experiment manifest records hashes of the A1 protocol, shared source, and
checkpoints for A3.

### Assignment 3: held-out inference

Import the same A1 and A2 modules, verify the saved hashes, reconstruct the exact
model configurations, and load the validation-selected checkpoints. Complete
held-out MNIST and EuroSAT inference and three-class Planet inference/GeoTIFF export. Test
results describe generalization; they must not be used to select a new model.

## Editing and checking work

The notebooks orchestrate the workflow: use them to read instructions, reload
modules, run checks, make figures, and write evidence and interpretations. The
assessed Python implementations go in these exact files:

| Assignment | File students edit | Symbols students implement |
|---|---|---|
| A1 | `src/student_pipeline.py` | `paired_flip`, `normalize_image`, `training_transform`, `evaluation_transform`, `build_mnist_loaders`, `build_eurosat_loaders`, `PlanetCatalogDataset`, `match_directory_pairs`, `PlanetDirectoryDataset`, `build_planet_loaders`, `save_data_protocol`, `load_data_protocol` |
| A2 | `src/student_models.py` | `MNISTResNet18`, `EuroSATClassifier`, `PlanetUNet` |
| A2 | `src/student_training.py` | `run_epoch`, `fit` |
| A3 | `src/student_inference.py` | `predict_classification`, `export_prediction`, `predict_and_export` |

Open those files in the JupyterLab or Colab file browser and replace the
`TODO`/`NotImplementedError` bodies while preserving the documented interfaces.
After saving, rerun the notebook reload cell and recreate affected objects. Add
written answers and required evidence in the labelled notebook Markdown cells,
and record agent use in `AI_USE.md`.

Do not edit `assignment_setup.py`, `student_checks.py`, `assessment_support.py`, the
catalog, or the tests. Supplied checks are feedback, not answers or proof of
scientific validity. Do not hide assessed implementations in notebook scratch
cells: A2 and A3 must be able to import the saved modules from earlier stages.

Optional terminal checks from this folder are:

```bash
PYTHONPATH=src python -m pytest tests
PYTHONPATH=src python -m pytest completion_tests
```

The starter tests pass before implementation. Completion tests are expected to
fail until the assessed functions are complete.

## Instructor grading command

The repository's `grading/` folder contains rubric-mapped behavioral checks but
no solutions. From an instructor copy, point it at a student's complete folder:

```bash
python grading/grade_assignment.py 1 --submission-root /path/to/student/assignments
python grading/grade_assignment.py 2 --submission-root /path/to/student/assignments
python grading/grade_assignment.py 3 --submission-root /path/to/student/assignments
```

Each command writes a Markdown and JSON report in `grading_reports/`. It awards
only the automatable subtotal and lists the remaining notebook evidence and
critical-question points for manual review. See `grading/README.md` for the safe,
standard correction workflow.

## Agentic coding

Coding agents may help with planning, one named TODO, debugging a failing check,
or reviewing code. Do not ask an agent to complete the entire assignment. Inspect
every suggested change, add an independent counterexample, and explain the final
implementation yourself. Record prompts, accepted/rejected suggestions, changes,
and verification in `AI_USE.md`.

## Submission and private material

Submit executed notebooks, shared source, `AI_USE.md`, protocols, small histories,
metrics, diagrams, figures, and rerun instructions. Keep raw data, environments,
and large checkpoints outside Git; provide approved retrieval details and hashes.

The instructor's ignored `answers/` folder is never included in the student ZIP.
Student distributions are built from an explicit allowlist, not by compressing a
working directory.

Each notebook begins with a 100-point grading rubric and explains the relevant
concepts, expected shapes, ordered implementation steps, checks, required evidence,
and interpretation before students reach the TODO cells.
