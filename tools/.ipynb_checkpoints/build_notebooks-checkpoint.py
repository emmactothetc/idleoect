"""Maintainer script that builds the three progressive student notebooks."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md(source):
    rendered = textwrap.dedent(source).strip() + "\n"
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": rendered.splitlines(keepends=True),
    }


def code(source):
    rendered = textwrap.dedent(source).strip() + "\n"
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": rendered.splitlines(keepends=True),
    }


def notebook(cells):
    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index:03d}"
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SETUP_MARKDOWN = r"""
## Setup: upload the complete folder

**2i2c is the primary platform.** Upload or extract the entire folder as
`/home/jovyan/assignments`, then open this notebook from the folder root. Keep
`src/`, `catalog.csv`, all notebooks, and `outputs/` together.
The `/home/jovyan` directory is persistent across normal 2i2c sessions.

The setup below adds `PROJECT_ROOT/src` directly to Python's import path. It does
not create an environment or install the assignment as a package. It first checks
the modules already supplied by the 2i2c image. Only if that check reports a
missing package should you open a terminal in `assignments` and run:

```bash
python -m pip install -r requirements.txt
```

**Colab fallback:** put the complete folder in
`My Drive/IDLEO/assignments`, open the notebook in Colab, and run the same
cells. Code, protocols, outputs, and checkpoints remain in Drive. Runtime data
may need to be downloaded again after a reset.
"""


SETUP_CODE = r"""
import importlib
import json  # noqa: F401
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    PROJECT_ROOT = Path("/content/drive/MyDrive/IDLEO/assignments")
else:
    preferred = Path("/home/jovyan/assignments")
    if preferred.is_dir():
        PROJECT_ROOT = preferred
    else:
        working_directory = Path.cwd().resolve()
        PROJECT_ROOT = working_directory

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from assignment_setup import check_dependencies, prepare_project  # noqa: E402

paths = prepare_project(PROJECT_ROOT)
versions = check_dependencies()
device_name = "cuda" if __import__("torch").cuda.is_available() else "cpu"
print("Project:", paths.project_root)
print("Data:", paths.data_root)
print("Device:", device_name)
print("Dependencies:", versions)
"""


AGENTIC = r"""
## Agentic coding without outsourcing the assignment

An agent may help you plan, inspect an error, propose a test, or review one saved
function. Work on one named TODO at a time. Give the function contract, shapes,
and the failing check; ask for an explanation of the proposed change. Read every
line, reject unsupported assumptions, save the file yourself, reload it, and run
both the supplied check and an independent counterexample. Do not ask an agent to
"complete the assignment" or paste an answer repository. Record what you accepted,
changed, rejected, and verified in `AI_USE.md`. You must be able to explain and
modify the final code without assistance.
"""


STUDENT_WORK_LOCATIONS = r"""
## Where you write code in the modular assignment

The notebooks are the **orchestrators**, not the main implementation files. Keep
them open to read instructions, reload your modules, run checks, make figures, and
write interpretations. Write assessed Python implementations in these exact files:

| Assignment | File you edit | Symbols you implement |
|---|---|---|
| A1 | `src/student_pipeline.py` | `paired_flip`, `normalize_image`, `training_transform`, `evaluation_transform`, `build_mnist_loaders`, `build_eurosat_loaders`, `PlanetCatalogDataset`, `match_directory_pairs`, `PlanetDirectoryDataset`, `build_planet_loaders`, `save_data_protocol`, `load_data_protocol` |
| A2 | `src/student_models.py` | `MNISTResNet18`, `EuroSATClassifier`, `PlanetUNet` |
| A2 | `src/student_training.py` | `run_epoch`, `fit` |
| A3 | `src/student_inference.py` | `predict_classification`, `export_prediction`, `predict_and_export` |

Replace the `TODO`/`NotImplementedError` bodies while preserving the public
function names, arguments, return contracts, and class attributes described in
their docstrings. Save the module, rerun the notebook's reload cell, and recreate
any datasets, loaders, models, or optimizers that were built from the old code.

You also write explanatory work in the notebook's clearly labelled evidence,
architecture, interpretation, and critical-question Markdown cells, and document
agent assistance in `AI_USE.md`. Do **not** implement assessed work in notebook
scratch cells or edit `assignment_setup.py`, `student_checks.py`, `assessment_support.py`,
the catalog, or the tests. Those supplied files define setup, checks, and evidence
contracts and should remain unchanged.
"""


A1_RUBRIC = r"""
## Grading rubric (100 points)

| Assessed evidence | Points | Full-credit standard |
|---|---:|---|
| MNIST transforms and reusable loaders | 20 | Correct normalization; seeded 90/10 development split; untouched official test set; train-only augmentation; exact IDs returned |
| EuroSAT100 loaders | 15 | Official 60/20/20 membership preserved; correct 13-band tensors; train-only shuffling; exact global IDs |
| Planet catalog pipeline | 30 | Lazy paired reads; four image bands; labels 0/1/2 and ignore -1; CRS/transform/grid checks; image-valid normalization; paired augmentation |
| Independent directory pairing | 15 | Site/date key matching; missing/ambiguous pairs rejected; site-aware development split; catalog agreement demonstrated |
| Reproducibility protocol and evidence | 10 | Seed, all memberships, preprocessing, class mapping, budgets, shapes, figures, and rerun evidence saved |
| Critical understanding answers | 10 | Specific, technically correct answers tied to observed evidence rather than generic definitions |
| **Total** | **100** | |

Partial credit is based on demonstrated behavior, not the number of lines written.
A loader that executes but leaks test data, misaligns masks, or discards class 2
cannot receive full credit. Required evidence must remain visible in the submitted
executed notebook.
"""


A1_FOUNDATIONS = r"""
## Foundations: what the data pipeline is responsible for

A PyTorch `Dataset` defines the meaning of one indexed sample. A `DataLoader`
controls how those samples become batches: ordering, shuffling, collation, worker
processes, and batch size. Keeping these responsibilities separate is important.
If a dataset silently chooses a different label or transform each time it is
constructed, later assignments cannot reproduce A1 even if their loader settings
look identical.

Three splits serve different purposes:

- **training** updates parameters and may use stochastic augmentation;
- **validation** supports development decisions and checkpoint selection;
- **test** is opened only after the full procedure is frozen.

A seed makes a randomized operation repeatable only when the same algorithm,
input membership, and library behavior are used. Therefore this assignment also
saves explicit IDs. The IDs—not the seed alone—are the durable handoff to A2/A3.

MNIST is the controlled introduction: one grayscale channel, fixed 28×28 pixels,
ten classes, and a packaged download. EuroSAT100 introduces 13 spectral bands and
an existing official split. Planet introduces the harder EO case: imagery and
categorical masks are separate georeferenced files, invalid pixels exist, and a
spatial transform must keep every image pixel aligned with its target pixel.

### Shapes to reason about before coding

| Dataset | One image | One target | One batch |
|---|---|---|---|
| MNIST | `[1,28,28]` float | scalar class 0–9 | `[B,1,28,28]`, `[B]` |
| EuroSAT100 | `[13,64,64]` float | scalar class 0–9 | `[B,13,64,64]`, `[B]` |
| Planet | `[4,H,W]` float | `[H,W]` class map | `[B,4,H,W]`, `[B,H,W]` |

Classification assigns one label to an image. Segmentation assigns one label to
each valid pixel. Do not add a singleton channel to an integer segmentation target
for cross entropy, and do not one-hot encode it unless an explicitly chosen loss
requires that representation.
"""


A2_RUBRIC = r"""
## Grading rubric (100 points)

| Assessed evidence | Points | Full-credit standard |
|---|---:|---|
| Exact reconstruction of A1 loaders/protocol | 10 | MNIST, EuroSAT100, and Planet membership equality asserted; no copied loader definitions |
| Adapted MNIST ResNet18 | 12 | One-channel input, ten raw logits, measured shape check, configuration saved |
| EuroSAT classifier | 10 | Thirteen-channel input, ten raw logits, measured shapes and architecture explanation |
| Five-stage Planet U-Net | 25 | Five encoders/downsamplings, separate bottleneck, five decoders/skips, three logits, even/odd size preservation |
| Training and validation implementation | 25 | Every batch processed; correct device/grad/mode handling; observation-weighted losses; multiclass metrics; validation-selected checkpoints |
| Histories, diagrams, shape tables, hashes | 8 | CSVs, curves, architecture diagram, measured table, configurations and hashes saved |
| Critical understanding answers | 10 | Correct reasoning about logits, receptive field, skips, imbalance, modes, validation, and checkpoints |
| **Total** | **100** | |

Code receives credit only with verification evidence. Hard-coded shapes, test-set
checkpoint selection, a two-stage U-Net, or binary Planet output lose the
corresponding rubric credit even when the notebook reaches its final cell.
"""


A2_FOUNDATIONS = r"""
## Foundations: from batches to trainable models

A model's final dimension describes the prediction problem. Image classifiers
return `[B,C]`; the Planet segmenter returns `[B,C,H,W]`. These are **raw logits**:
unbounded scores that cross entropy converts internally with log-softmax. Applying
softmax before `CrossEntropyLoss` changes the optimization problem and should not
be done here.

The MNIST ResNet18 exercise shows how to adapt a supplied architecture: its first
convolution must accept one channel, the early max-pool is unnecessarily aggressive
for 28×28 digits, and its final linear layer must produce ten logits. EuroSAT
requires thirteen input channels. Neither modification changes the meaning of a
batch target: it remains one integer class per image.

The Planet U-Net is symmetric but not shallow. Each of five encoder blocks creates
a skip tensor **before** pooling. The separate bottleneck operates after the fifth
downsampling, at about 1/32 input resolution. Five decoder stages progressively
upsample and fuse matching skips. Deep features carry broader context; skips restore
location and boundary detail. Odd image sizes require explicit alignment to the
measured skip size rather than assumptions based on exact powers of two.

One epoch means every batch in one loader has been processed once. Training uses
`train()`, gradients, `zero_grad`, `backward`, and `step`. Validation uses `eval()`
and `no_grad()` with no optimizer update. Average loss must be weighted by the
number of samples or valid pixels; an unweighted average of unequal batch means is
biased. The minimum validation loss selects a checkpoint. The test split is not a
checkpoint-selection tool.
"""


A3_RUBRIC = r"""
## Grading rubric (100 points)

| Assessed evidence | Points | Full-credit standard |
|---|---:|---|
| Provenance and exact reconstruction | 15 | A1 protocol, A2 source, constructor configuration, and all checkpoint hashes verified before inference |
| Held-out MNIST inference | 10 | Exact A1 test membership; exact A2 ResNet checkpoint; every batch; confusion/per-class evidence |
| Held-out EuroSAT inference | 10 | Exact A1 test membership; exact A2 classifier checkpoint; every batch and per-class support |
| Planet inference and GeoTIFF export | 25 | Three-class argmax; every test chip; global metrics; class 2 retained; CRS/transform/grid/dtype/nodata independently verified |
| Failure analysis and model card | 25 | Predetermined ranking; weak and median cases; per-class interpretation; limitations, provenance and prohibited uses documented |
| Critical understanding answers | 15 | Specific reasoning about leakage, undefined metrics, spatial dependence, nodata, accuracy and failure cases |
| **Total** | **100** | |

Inference credit requires a frozen model and test set. Retraining, changing
preprocessing, or selecting a checkpoint after inspecting test outcomes invalidates
the affected evaluation evidence.
"""


A3_FOUNDATIONS = r"""
## Foundations: inference is a provenance exercise

Loading a `.pth` file is not sufficient evidence that the intended model ran. A
`state_dict` stores named tensors, not the Python forward graph, preprocessing, or
data membership. A3 therefore rebuilds the saved constructor configuration, checks
source and checkpoint hashes, and reconstructs A1's exact test IDs before loading
weights. If any hash differs, stop and explain the discrepancy rather than editing
the manifest.

Inference uses `eval()` and `no_grad()` for every batch. For classification,
`argmax(dim=1)` converts ten logits to one class ID. For segmentation it converts
`[B,3,H,W]` to `[B,H,W]` while preserving boundary class 2. Metrics are accumulated
from one global confusion matrix so a small last batch or a chip with few valid
pixels is not accidentally weighted like a large batch.

Pixel accuracy alone is unsafe for imbalanced maps. Report class support, IoU,
Dice, precision, recall, and the confusion matrix. A zero denominator means the
metric is undefined, not automatically zero. Georeferenced output adds another
contract: prediction height/width, CRS, affine transform, dtype, class values, and
nodata must be verified after reopening the file. Image validity controls output
nodata; missing reference annotation controls only which pixels contribute to
evaluation.

Failure analysis is not a gallery of attractive examples. Freeze a ranking rule
before looking at images, include weak cases, compare interior and boundary errors,
and distinguish an observed failure from a causal claim. The model card records
what was run, where it may fail, and uses that are not justified by this exercise.
"""


def assignment_1():
    return notebook(
        [
            md(
                r"""
                # Assignment 1: Build a reusable Earth-observation data pipeline

                This is the first stage of one progressive workflow. You will implement
                `src/student_pipeline.py`, save exact dataset membership and preprocessing
                settings, and reuse that module and protocol unchanged in Assignments 2 and 3.

                | Task | Student work | Evidence |
                |---|---|---|
                | MNIST | packaged grayscale loaders | normalization, augmentation, saved 90/10/test IDs |
                | EuroSAT100 | documented 60/20/20 loaders | shapes, labels, RGB panels, saved global IDs |
                | Planet catalog | lazy four-band/three-class dataset | grid/validity checks, class-2 preservation |
                | Planet directories | independent site/date pairing | missing/ambiguous-pair tests and agreement |
                | Protocol | portable JSON | exact seed, membership, transforms and budgets |
                """
            ),
            md(A1_RUBRIC),
            md(STUDENT_WORK_LOCATIONS),
            md(A1_FOUNDATIONS),
            md(SETUP_MARKDOWN),
            code(SETUP_CODE),
            md(AGENTIC),
            md(
                r"""
                ## How to complete source-file tasks

                Open `src/student_pipeline.py` in the JupyterLab file browser. Read each
                complete docstring, replace only its `NotImplementedError`, and save.
                Rerun the reload cell after every edit. Existing dataset objects retain old
                class definitions, so recreate them after reloading.
                """
            ),
            code(
                r"""
                import matplotlib.pyplot as plt
                import torch
                import student_pipeline as pipeline
                import student_checks as checks
                from assessment_support import fetch_planet_pairs, set_seed, show_eurosat_batch, show_planet_triplet

                pipeline = importlib.reload(pipeline)
                checks = importlib.reload(checks)
                set_seed(42)
                """
            ),
            md(
                r"""
                ## Choose the data budget before inspecting outcomes

                These small values make setup and debugging possible on CPU. Increase them
                only under the course assessment instructions. Record every change before
                looking at validation or test results.
                """
            ),
            code(
                r"""
                SEED = 42
                MNIST_LIMITS = {"train": 512, "validate": 128, "test": 128}
                EUROSAT_LIMITS = {"train": 60, "validate": 20, "test": 20}
                PLANET_LIMITS = {"train": 8, "validate": 4, "test": 4}
                MNIST_BATCH_SIZE = 64
                EURO_BATCH_SIZE = 16
                PLANET_BATCH_SIZE = 2
                DOWNLOAD_MISSING_PLANET = True
                """
            ),
            md(
                r"""
                ## Task 1: MNIST transforms, split, loaders, and visualization (20 points)

                Implement `build_mnist_loaders` before working with EO rasters. Start from
                the packaged 60,000-image training pool and official 10,000-image test pool.
                Use the seed to create a 90/10 training/validation split, but save the exact
                selected global IDs. Training may rotate digits; validation/test may not.
                Normalize all splits with mean `0.1307` and standard deviation `0.3081`.

                **Ordered implementation:** (1) define separate train/evaluation transforms;
                (2) construct the packaged datasets; (3) generate or replay membership;
                (4) create three loaders with only training shuffled; (5) return loaders and
                IDs; (6) verify shapes, normalized values, labels, disjointness, and repeated
                evaluation batches. The limits below are a runtime budget, not a new split.
                """
            ),
            code(
                r"""
                mnist = pipeline.build_mnist_loaders(
                    paths.data_root / "mnist",
                    batch_size=MNIST_BATCH_SIZE,
                    seed=SEED,
                    limits=MNIST_LIMITS,
                    download=True,
                )
                mnist_images, mnist_labels = next(iter(mnist["train"]))
                print("MNIST:", mnist_images.shape, mnist_labels.shape)
                print("membership sizes:", {key: len(value) for key, value in mnist["split_ids"].items()})
                assert mnist_images.shape[1:] == (1, 28, 28)
                assert set(mnist["split_ids"]["train"]).isdisjoint(mnist["split_ids"]["validate"])
                figure, axes = plt.subplots(2, 4, figsize=(8, 4))
                for axis, image, label in zip(axes.flat, mnist_images[:8], mnist_labels[:8]):
                    axis.imshow(image[0].cpu(), cmap="gray")
                    axis.set_title(f"label {int(label)}")
                    axis.axis("off")
                figure.tight_layout()
                figure.savefig(paths.output_root / "assignment_1/mnist_batch.png", dpi=150)
                """
            ),
            md(
                r"""
                **Evidence to submit (20 points):** implementation, transform explanation,
                membership counts and overlap assertions, tensor/label shapes, eight-image
                figure, and a short interpretation of values after normalization. Explain
                why the official test pool is not split or inspected during development.
                """
            ),
            md(
                r"""
                ## Task 2: EuroSAT100 loaders (15 points)

                Implement `build_eurosat_loaders`. Preserve EuroSAT100's documented
                60/20/20 split and return its exact global integer membership. Shuffle only training. Explain the
                `[B,13,H,W]` image and `[B]` label shapes and why validation/test membership
                must not be regenerated in later assignments.
                """
            ),
            code(
                r"""
                eurosat = pipeline.build_eurosat_loaders(
                    paths.data_root / "eurosat",
                    batch_size=EURO_BATCH_SIZE,
                    seed=SEED,
                    limits=EUROSAT_LIMITS,
                    download=True,
                )
                euro_images, euro_labels = next(iter(eurosat["train"]))
                print(euro_images.shape, euro_labels.shape, eurosat["split_ids"])
                euro_figure = show_eurosat_batch(euro_images, euro_labels)
                euro_figure.savefig(paths.output_root / "assignment_1/eurosat_batch.png", dpi=150)
                """
            ),
            md(
                r"""
                **Written evidence:** describe the split, preprocessing, channel meanings,
                tensor shapes, and one check showing train/validate/test IDs are disjoint.
                """
            ),
            md(
                r"""
                ## Task 3: Planet catalog dataset and loaders (30 points)

                Implement paired transforms, image-only normalization,
                `PlanetCatalogDataset`, and `build_planet_loaders`. Planet masks retain
                `0=noncrop`, `1=field interior`, `2=boundary`; invalid targets are `-1`.
                Verify image/mask CRS, affine transform, and grid before accepting a pair.
                Training may use paired stochastic flips; validation and test must remain
                deterministic. Never use test labels to choose preprocessing or models.
                """
            ),
            code(
                r"""
                import pandas as pd

                planet = pipeline.build_planet_loaders(
                    paths.catalog_path,
                    paths.data_root,
                    batch_size=PLANET_BATCH_SIZE,
                    seed=SEED,
                    limits=PLANET_LIMITS,
                )
                catalog = pd.read_csv(paths.catalog_path)
                selected_indices = sum(planet["selected_rows"].values(), [])
                selected_catalog = catalog.iloc[selected_indices]
                if DOWNLOAD_MISSING_PLANET:
                    print(fetch_planet_pairs(selected_catalog, paths.data_root))

                pipeline = importlib.reload(pipeline)
                planet = pipeline.build_planet_loaders(
                    paths.catalog_path,
                    paths.data_root,
                    batch_size=PLANET_BATCH_SIZE,
                    seed=SEED,
                    selected_rows=planet["selected_rows"],
                )
                planet_images, planet_targets, planet_sources = next(iter(planet["train"]))
                print(planet_images.shape, planet_targets.shape, torch.unique(planet_targets))
                planet_figure = show_planet_triplet(planet_images[0], planet_targets[0])
                planet_figure.savefig(paths.output_root / "assignment_1/planet_pair.png", dpi=150)
                checks.check_transforms(pipeline)
                checks.check_loaders(eurosat, planet, mnist)
                """
            ),
            md(
                r"""
                **Independent verification:** add at least two counterexamples, such as an
                asymmetric marker proving paired flips, a constant band with invalid pixels,
                or a raster pair with mismatched transform. A loader that runs can still be
                scientifically wrong.
                """
            ),
            md(
                r"""
                ## Task 4: Independent directory pairing (15 points)

                Implement `match_directory_pairs` and `PlanetDirectoryDataset`. Match image
                and mask by site and acquisition date, not equal filename stems. Use only
                selected train+validate image names, reject missing/ambiguous masks, keep all
                dates from one site in the same development split, and compare one
                unaugmented directory sample with the catalog dataset.
                """
            ),
            code(
                r"""
                from pathlib import Path

                development_rows = selected_catalog[selected_catalog.split.isin(["train", "validate"])]
                allowed_names = [Path(value).name for value in development_rows.window_b]
                pairs = pipeline.match_directory_pairs(
                    paths.data_root / "mappingafrica-256/images",
                    paths.data_root / "mappingafrica-256/labels",
                    allowed_names,
                )
                sites = sorted({image.stem.split("_")[0] for image, _ in pairs})
                generator = torch.Generator().manual_seed(SEED)
                order = torch.randperm(len(sites), generator=generator).tolist()
                train_sites = {sites[index] for index in order[: max(1, int(0.8 * len(sites)))]}
                train_pairs = [pair for pair in pairs if pair[0].stem.split("_")[0] in train_sites]
                validation_pairs = [pair for pair in pairs if pair[0].stem.split("_")[0] not in train_sites]
                directory_train = pipeline.PlanetDirectoryDataset(train_pairs, pipeline.training_transform)
                directory_validate = pipeline.PlanetDirectoryDataset(validation_pairs, pipeline.evaluation_transform)
                print("directory train/validate:", len(directory_train), len(directory_validate))
                """
            ),
            md(
                r"""
                ## Task 5: Save the protocol for A2 and A3 (10 points)

                This file is the handoff between assignments. It must contain relative or
                integer identifiers—not machine-specific absolute paths.
                """
            ),
            code(
                r"""
                protocol_path = paths.output_root / "assignment_1/data_protocol.json"
                pipeline.save_data_protocol(
                    protocol_path,
                    seed=SEED,
                    mnist_split_ids=mnist["split_ids"],
                    eurosat_split_ids=eurosat["split_ids"],
                    planet_selected_rows=planet["selected_rows"],
                    settings={
                        "mnist_batch_size": MNIST_BATCH_SIZE,
                        "euro_batch_size": EURO_BATCH_SIZE,
                        "planet_batch_size": PLANET_BATCH_SIZE,
                        "planet_classes": [0, 1, 2],
                        "planet_preprocessing": (
                            "per-image band z-score over image-valid pixels"
                        ),
                    },
                )
                print(protocol_path, protocol_path.read_text())
                """
            ),
            md(
                r"""
                ## Critical understanding questions (10 points)

                Answer in your own words and refer to evidence from this notebook.

                1. How could sharing one mutable transform object between training and validation cause leakage or nondeterminism?
                2. Why is replaying explicit MNIST IDs stronger evidence than merely reusing seed 42?
                3. Why must the same spatial augmentation be applied to imagery and categorical masks?
                4. Why does matching image and mask array shape not prove they describe the same geographic grid?
                5. Why must missing reference labels not change image normalization or inference validity?
                6. What information can per-image normalization remove from multispectral imagery?
                7. Why do non-overlapping catalog IDs not prove geographic independence?
                8. Which exact artifact ensures A2 and A3 use the same examples as A1, and how would you verify it?
                9. What evidence would reveal that class `2` boundaries had accidentally been converted to binary foreground?
                10. Contrast one-image classification targets with per-pixel segmentation targets and their batch shapes.
                """
            ),
        ]
    )


def assignment_2():
    return notebook(
        [
            md(
                r"""
                # Assignment 2: Construct and train progressive EO models

                Do not copy dataloader code into this notebook. Import the A1 module and
                reconstruct the exact saved membership from `data_protocol.json`.

                | Task | Student work | Evidence |
                |---|---|---|
                | Reuse A1 | rebuild exact loaders | protocol and membership equality |
                | MNIST | adapt ResNet18 for 1×28×28 inputs | logits, history, validation checkpoint |
                | EuroSAT | implement 13-band classifier | architecture/shape table and curves |
                | Planet | implement five-stage, three-class U-Net | five skips, bottleneck, diagram |
                | Training | complete full train/validation loops | histories and best checkpoints |
                """
            ),
            md(A2_RUBRIC),
            md(STUDENT_WORK_LOCATIONS),
            md(A2_FOUNDATIONS),
            md(SETUP_MARKDOWN),
            code(SETUP_CODE),
            md(AGENTIC),
            md("## Reload the saved A1 and A2 source modules"),
            code(
                r"""
                import matplotlib.pyplot as plt
                import pandas as pd
                import torch
                import student_pipeline as pipeline
                import student_models as models
                import student_training as training
                import student_checks as checks
                from assessment_support import sha256, set_seed

                pipeline = importlib.reload(pipeline)
                models = importlib.reload(models)
                training = importlib.reload(training)
                set_seed(42)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                """
            ),
            md(
                r"""
                ## Task 1: Reconstruct A1 data exactly (10 points)

                Load the saved protocol and pass its IDs into the A1 constructors. Assert
                exact equality before training. Repairing A1 preprocessing creates a new
                experiment: rerun A1 and save a new protocol rather than silently changing it.
                """
            ),
            code(
                r"""
                protocol_path = paths.output_root / "assignment_1/data_protocol.json"
                protocol = pipeline.load_data_protocol(protocol_path)
                if "mnist_split_ids" not in protocol:
                    raise ValueError("A1 protocol predates the required MNIST track; rerun A1")
                mnist = pipeline.build_mnist_loaders(
                    paths.data_root / "mnist",
                    batch_size=protocol["settings"]["mnist_batch_size"],
                    seed=protocol["seed"],
                    split_ids=protocol["mnist_split_ids"],
                    download=True,
                )
                eurosat = pipeline.build_eurosat_loaders(
                    paths.data_root / "eurosat",
                    batch_size=protocol["settings"]["euro_batch_size"],
                    seed=protocol["seed"],
                    split_ids=protocol["eurosat_split_ids"],
                    download=True,
                )
                planet = pipeline.build_planet_loaders(
                    paths.catalog_path,
                    paths.data_root,
                    batch_size=protocol["settings"]["planet_batch_size"],
                    seed=protocol["seed"],
                    selected_rows=protocol["planet_selected_rows"],
                )
                assert mnist["split_ids"] == protocol["mnist_split_ids"]
                assert eurosat["split_ids"] == protocol["eurosat_split_ids"]
                assert planet["selected_rows"] == protocol["planet_selected_rows"]
                """
            ),
            md(
                r"""
                ## Task 2: Adapt ResNet18 for MNIST (12 points)

                Implement `MNISTResNet18` in `student_models.py` using
                `torchvision.models.resnet18(weights=None)`. Replace the first convolution
                so it accepts one channel, replace the early max-pool with `Identity` for
                small 28×28 inputs, and replace the final layer with ten logits. Do not
                download pretrained weights. Record the input, intermediate/output shape,
                trainable parameter count, and why logits—not probabilities—are returned.
                """
            ),
            code(
                r"""
                mnist_config = {"num_classes": 10}
                mnist_classifier = models.MNISTResNet18(**mnist_config).to(device)
                mnist_images, mnist_targets = next(iter(mnist["train"]))
                mnist_logits = mnist_classifier(mnist_images.to(device))
                print("MNIST:", mnist_images.shape, mnist_logits.shape)
                print("parameters:", sum(parameter.numel() for parameter in mnist_classifier.parameters()))
                assert mnist_logits.shape == (mnist_images.shape[0], 10)
                """
            ),
            md(
                r"""
                **Evidence to submit (12 points):** the adapted layers, measured shape and
                parameter count, and a concise explanation of why an unchanged three-channel
                ImageNet stem is incompatible with MNIST.
                """
            ),
            md(
                r"""
                ## Task 3: EuroSAT classifier (10 points)

                Implement `EuroSATClassifier` in `student_models.py`. It accepts 13-band
                `[B,13,H,W]` tensors and returns ten raw logits `[B,10]`. Draw the model and
                record measured feature shapes rather than guessing them.
                """
            ),
            code(
                r"""
                classifier_config = {"in_channels": 13, "num_classes": 10, "base_channels": 32}
                classifier = models.EuroSATClassifier(**classifier_config).to(device)
                euro_images, euro_targets = next(iter(eurosat["train"]))
                print("EuroSAT:", euro_images.shape, classifier(euro_images.to(device)).shape)
                """
            ),
            md(
                r"""
                ## Task 4: Five-stage Planet U-Net (25 points)

                Implement `PlanetUNet`, not the earlier two-stage teaching baseline. It must
                expose five encoder stages, a separate bottleneck, five decoder stages, and
                five corresponding skip connections. A four-band input returns three raw
                class logits per pixel. Preserve even and odd input sizes. Submit a readable
                architecture diagram labelling stages, skips, channels, and spatial sizes.
                """
            ),
            code(
                r"""
                unet_config = {"in_channels": 4, "num_classes": 3, "base_channels": 8}
                segmenter = models.PlanetUNet(**unet_config).to(device)
                checks.check_unet_architecture(segmenter.cpu())
                segmenter = segmenter.to(device)
                planet_images, planet_targets, _ = next(iter(planet["train"]))
                planet_logits = segmenter(planet_images.to(device))
                print("Planet:", planet_images.shape, planet_logits.shape, planet_targets.shape)
                assert planet_logits.shape == (planet_images.shape[0], 3, *planet_targets.shape[-2:])
                """
            ),
            md(
                r"""
                **Architecture and evidence (8 points):** add your own diagram and a measured table showing
                all five encoder outputs, the 1/32-resolution bottleneck, and five decoder
                outputs. Explain how each skip is aligned for odd-sized inputs.
                """
            ),
            md(
                r"""
                ## Task 5: Full training and validation loops (25 points)

                Implement `run_epoch` and `fit` in `student_training.py`. You own every batch
                iteration, device transfer, gradient operation, validation context, loss
                aggregation, global confusion count, history row, and best-validation-loss
                checkpoint. Segmentation uses three-class cross entropy with target `-1`
                ignored. Test loaders are prohibited here.
                """
            ),
            code(
                r"""
                checks.check_training_loop(training, device)
                EPOCHS = 3
                MNIST_EPOCHS = 2

                mnist_optimizer = torch.optim.Adam(mnist_classifier.parameters(), lr=1e-3)
                mnist_checkpoint = paths.output_root / "assignment_2/mnist_resnet18.pth"
                mnist_history = training.fit(
                    mnist_classifier,
                    mnist["train"],
                    mnist["validate"],
                    device,
                    "classification",
                    mnist_optimizer,
                    MNIST_EPOCHS,
                    mnist_checkpoint,
                    mnist_config,
                )
                mnist_history.to_csv(paths.output_root / "assignment_2/mnist_history.csv", index=False)

                classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
                classifier_checkpoint = paths.output_root / "assignment_2/eurosat_classifier.pth"
                classifier_history = training.fit(
                    classifier,
                    eurosat["train"],
                    eurosat["validate"],
                    device,
                    "classification",
                    classifier_optimizer,
                    EPOCHS,
                    classifier_checkpoint,
                    classifier_config,
                )
                classifier_history.to_csv(paths.output_root / "assignment_2/eurosat_history.csv", index=False)

                segmentation_optimizer = torch.optim.Adam(segmenter.parameters(), lr=1e-3)
                segmentation_checkpoint = paths.output_root / "assignment_2/planet_unet.pth"
                segmentation_history = training.fit(
                    segmenter,
                    planet["train"],
                    planet["validate"],
                    device,
                    "segmentation",
                    segmentation_optimizer,
                    EPOCHS,
                    segmentation_checkpoint,
                    unet_config,
                )
                segmentation_history.to_csv(paths.output_root / "assignment_2/planet_history.csv", index=False)
                display(mnist_history, classifier_history, segmentation_history)

                figure, axes = plt.subplots(1, 3, figsize=(15, 4))
                for axis, history, title in zip(
                    axes,
                    (mnist_history, classifier_history, segmentation_history),
                    ("MNIST", "EuroSAT", "Planet"),
                ):
                    history.plot(x="epoch", y=["train_loss", "validation_loss"], ax=axis)
                    axis.set_title(title)
                figure.tight_layout()
                figure.savefig(paths.output_root / "assignment_2/training_curves.png", dpi=150)
                """
            ),
            md(
                r"""
                **Experiment evidence:** plot train/validation loss, report wall time and
                budgets, compare against majority/noncrop baselines, and interpret per-class
                support and IoU. Diagnose a failed or inconclusive experiment honestly; a
                declining loss alone does not establish useful boundary segmentation.
                """
            ),
            md("## Save the progressive handoff manifest"),
            code(
                r"""
                manifest = {
                    "data_protocol_sha256": sha256(protocol_path),
                    "student_pipeline_sha256": sha256(paths.project_root / "src/student_pipeline.py"),
                    "student_models_sha256": sha256(paths.project_root / "src/student_models.py"),
                    "student_training_sha256": sha256(paths.project_root / "src/student_training.py"),
                    "mnist_checkpoint": mnist_checkpoint.name,
                    "mnist_checkpoint_sha256": sha256(mnist_checkpoint),
                    "classifier_checkpoint": classifier_checkpoint.name,
                    "classifier_checkpoint_sha256": sha256(classifier_checkpoint),
                    "segmentation_checkpoint": segmentation_checkpoint.name,
                    "segmentation_checkpoint_sha256": sha256(segmentation_checkpoint),
                    "mnist_config": mnist_config,
                    "classifier_config": classifier_config,
                    "unet_config": unet_config,
                }
                manifest_path = paths.output_root / "assignment_2/experiment_manifest.json"
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                print(manifest_path.read_text())
                """
            ),
            md(
                r"""
                ## Critical understanding questions (10 points)

                1. Why does A2 import and reconstruct A1 loaders instead of copying their code?
                2. Why are ResNet18's input convolution and early max-pool both reconsidered for 28×28 one-channel digits?
                3. At which resolutions are the five U-Net skip tensors created, and why are they taken before pooling?
                4. What does the bottleneck add beyond simply using five encoder blocks?
                5. Why must the final Planet tensor be `[B,3,H,W]` rather than `[B,H,W]` or `[B,1,H,W]`?
                6. Why does cross entropy receive raw logits rather than softmax probabilities?
                7. How can background-dominated pixel accuracy conceal failure on class `2` boundaries?
                8. What state changes between `model.train()` and `model.eval()`, even when gradients are disabled?
                9. Why must the checkpoint be selected by validation loss rather than the final epoch or test score?
                10. How would you detect an incorrectly weighted mean of unequal batch losses?
                """
            ),
        ]
    )


def assignment_3():
    return notebook(
        [
            md(
                r"""
                # Assignment 3: Reconstruct, evaluate, and export

                A3 adds inference only. It imports A1's pipeline and A2's model/training
                modules, reconstructs exact saved architectures, verifies hashes, and uses
                held-out test loaders that were never used for selection.
                """
            ),
            md(A3_RUBRIC),
            md(STUDENT_WORK_LOCATIONS),
            md(A3_FOUNDATIONS),
            md(SETUP_MARKDOWN),
            code(SETUP_CODE),
            md(AGENTIC),
            md("## Reload the progressive source modules"),
            code(
                r"""
                import pandas as pd
                import torch
                import student_pipeline as pipeline
                import student_models as models
                import student_training as training
                import student_inference as inference
                import student_checks as checks
                from assessment_support import sha256, show_planet_triplet

                pipeline = importlib.reload(pipeline)
                models = importlib.reload(models)
                training = importlib.reload(training)
                inference = importlib.reload(inference)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                """
            ),
            md(
                r"""
                ## Task 1: Verify provenance and rebuild the exact workflow (part of 15 points)

                Load the A1 protocol and A2 manifest. Verify hashes before loading weights.
                Reconstruct dataloaders from saved membership and models from saved constructor
                dictionaries. Do not substitute a new architecture or new split.
                """
            ),
            code(
                r"""
                protocol_path = paths.output_root / "assignment_1/data_protocol.json"
                manifest_path = paths.output_root / "assignment_2/experiment_manifest.json"
                protocol = pipeline.load_data_protocol(protocol_path)
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                assert sha256(protocol_path) == manifest["data_protocol_sha256"]
                assert sha256(paths.project_root / "src/student_pipeline.py") == manifest["student_pipeline_sha256"]
                assert sha256(paths.project_root / "src/student_models.py") == manifest["student_models_sha256"]
                assert sha256(paths.project_root / "src/student_training.py") == manifest["student_training_sha256"]

                mnist = pipeline.build_mnist_loaders(
                    paths.data_root / "mnist",
                    batch_size=protocol["settings"]["mnist_batch_size"],
                    seed=protocol["seed"],
                    split_ids=protocol["mnist_split_ids"],
                    download=True,
                )
                eurosat = pipeline.build_eurosat_loaders(
                    paths.data_root / "eurosat",
                    batch_size=protocol["settings"]["euro_batch_size"],
                    seed=protocol["seed"],
                    split_ids=protocol["eurosat_split_ids"],
                    download=True,
                )
                planet = pipeline.build_planet_loaders(
                    paths.catalog_path,
                    paths.data_root,
                    batch_size=protocol["settings"]["planet_batch_size"],
                    seed=protocol["seed"],
                    selected_rows=protocol["planet_selected_rows"],
                )
                """
            ),
            md(
                "## Task 2: Restore validation-selected checkpoints (part of the same 15 points)"
            ),
            code(
                r"""
                mnist_path = paths.output_root / "assignment_2" / manifest["mnist_checkpoint"]
                classifier_path = paths.output_root / "assignment_2" / manifest["classifier_checkpoint"]
                segmenter_path = paths.output_root / "assignment_2" / manifest["segmentation_checkpoint"]
                assert sha256(mnist_path) == manifest["mnist_checkpoint_sha256"]
                assert sha256(classifier_path) == manifest["classifier_checkpoint_sha256"]
                assert sha256(segmenter_path) == manifest["segmentation_checkpoint_sha256"]

                mnist_classifier = models.MNISTResNet18(**manifest["mnist_config"]).to(device)
                mnist_state = torch.load(mnist_path, map_location=device)
                mnist_classifier.load_state_dict(mnist_state["model_state"])

                classifier = models.EuroSATClassifier(**manifest["classifier_config"]).to(device)
                classifier_state = torch.load(classifier_path, map_location=device)
                classifier.load_state_dict(classifier_state["model_state"])

                segmenter = models.PlanetUNet(**manifest["unet_config"]).to(device)
                segmenter_state = torch.load(segmenter_path, map_location=device)
                segmenter.load_state_dict(segmenter_state["model_state"])
                checks.check_unet_architecture(segmenter.cpu())
                segmenter = segmenter.to(device)
                """
            ),
            md(
                r"""
                ## Task 3: Held-out MNIST and EuroSAT inference (10 + 10 points)

                Implement `predict_classification`. Process every test batch under
                `eval()` and `no_grad()`, aggregate one confusion matrix, and report support
                and per-class metrics. A3 may inspect test errors after model selection but
                must not return to A2 and choose a different model from them.
                """
            ),
            code(
                r"""
                mnist_result = inference.predict_classification(mnist_classifier, mnist["test"], device)
                classification_result = inference.predict_classification(classifier, eurosat["test"], device)
                print("MNIST metrics:", mnist_result["metrics"])
                print("EuroSAT metrics:", classification_result["metrics"])
                display(pd.DataFrame({
                    "target": classification_result["targets"],
                    "prediction": classification_result["predictions"],
                }))
                print(classification_result["metrics"])
                """
            ),
            md(
                r"""
                ## Task 4: Three-class Planet inference and GeoTIFF export (25 points)

                Implement `export_prediction` and `predict_and_export`. Use `argmax` across
                three logits, preserve class `2`, and aggregate metrics across the entire
                test split. Each output must preserve source CRS, transform, height and
                width; use uint8 classes 0/1/2 and nodata 255 based on image validity—not
                missing reference labels. Reopen and validate every exported file.
                """
            ),
            code(
                r"""
                checks.check_export(inference, paths.output_root / "assignment_3")
                prediction_dir = paths.output_root / "assignment_3/predictions"
                segmentation_result = inference.predict_and_export(segmenter, planet["test"], device, prediction_dir)
                print(segmentation_result["metrics"])
                display(pd.DataFrame(segmentation_result["files"]))

                sample_images, sample_targets, _ = next(iter(planet["test"]))
                with torch.no_grad():
                    sample_predictions = segmenter(sample_images.to(device)).argmax(1).cpu()
                prediction_figure = show_planet_triplet(
                    sample_images[0], sample_targets[0], sample_predictions[0]
                )
                prediction_figure.savefig(
                    paths.output_root / "assignment_3/planet_prediction.png", dpi=150
                )
                """
            ),
            md(
                r"""
                ## Task 5: Failure analysis and model card (25 points)

                Before ranking examples, state a selection rule such as three lowest
                per-chip macro-IoU cases plus one median case, with filename tie-breaking.
                Compare noncrop, field-interior, and boundary behavior separately. Save a
                `model_card.md` describing purpose, data populations, splits, preprocessing,
                architecture, checkpoints, compute, metrics, limitations, and prohibited uses.

                **Written answer:** add the rule, measured results, failure examples, and
                model card here. Do not present only attractive maps.
                """
            ),
            code(
                r"""
                ranked_chips = sorted(
                    segmentation_result["files"],
                    key=lambda record: (record["metrics"]["mean_iou"], record["source_path"]),
                )
                failure_analysis = {
                    "selection_rule": (
                        "three lowest per-chip mean-IoU cases plus the median; "
                        "filename breaks ties"
                    ),
                    "lowest_three": ranked_chips[:3],
                    "median": ranked_chips[len(ranked_chips) // 2],
                }
                failure_path = paths.output_root / "assignment_3/failure_analysis.json"
                failure_path.write_text(json.dumps(failure_analysis, indent=2), encoding="utf-8")
                print(failure_path)
                """
            ),
            code(
                r'''MODEL_CARD = """# Model card: IDLEO 2026 reference models

## Intended purpose
TODO: describe the supported educational or research use and prohibited uses.

## Data and protocol
TODO: cite the saved A1 protocol, populations, split construction, and
preprocessing.

## Architecture and checkpoints
TODO: describe both architectures and record the exact checkpoint/source hashes.

## Evaluation
TODO: report aggregate and per-class metrics with denominators and uncertainty
caveats.

## Limitations and failure modes
TODO: discuss geographic dependence, boundary errors, coverage, and deployment
risks.
"""
model_card_path = paths.output_root / "assignment_3/model_card.md"
model_card_path.write_text(MODEL_CARD, encoding="utf-8")
print(model_card_path)'''
            ),
            code(
                r"""
                results_path = paths.output_root / "assignment_3/evaluation_summary.json"
                results_path.write_text(json.dumps({
                    "mnist_metrics": mnist_result["metrics"],
                    "classification_metrics": classification_result["metrics"],
                    "segmentation_metrics": segmentation_result["metrics"],
                    "prediction_files": segmentation_result["files"],
                    "failure_analysis": str(failure_path),
                    "data_protocol_sha256": sha256(protocol_path),
                    "experiment_manifest_sha256": sha256(manifest_path),
                }, indent=2), encoding="utf-8")
                print(results_path)
                """
            ),
            md(
                r"""
                ## Critical understanding questions (15 points)

                1. Which hashes prove that A3 used A1's protocol and A2's exact source/checkpoints?
                2. Why does the MNIST test loader have to replay A1 membership instead of simply requesting any test subset?
                3. Why is rebuilding the same class definition necessary before loading a `state_dict`?
                4. What information would be leaked if A3 test scores were used to choose a new checkpoint?
                5. Why should unsupported per-class metrics remain undefined rather than being replaced with zero?
                6. How can global pixel accuracy remain high while boundary IoU is poor?
                7. Why must GeoTIFF nodata follow image validity rather than reference-label availability?
                8. Why does removing duplicate IDs not establish geographic independence?
                9. How does a predetermined failure-example rule reduce presentation bias?
                10. Which observed limitation should motivate the next experiment without changing this test report?
                """
            ),
        ]
    )


def main():
    outputs = {
        "assignment1.ipynb": assignment_1(),
        "assignment2.ipynb": assignment_2(),
        "assignment3.ipynb": assignment_3(),
    }
    for name, content in outputs.items():
        path = ROOT / name
        path.write_text(json.dumps(content, indent=1), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
