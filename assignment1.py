#!/usr/bin/env python
# coding: utf-8

# # Assignment 1: Build a reusable Earth-observation data pipeline
# 
# This is the first stage of one progressive workflow. You will implement
# `src/student_pipeline.py`, save exact dataset membership and preprocessing
# settings, and reuse that module and protocol unchanged in Assignments 2 and 3.
# 
# | Task | Student work | Evidence |
# |---|---|---|
# | MNIST | packaged grayscale loaders | normalization, augmentation, saved 90/10/test IDs |
# | EuroSAT100 | documented 60/20/20 loaders | shapes, labels, RGB panels, saved global IDs |
# | Planet catalog | lazy four-band/three-class dataset | grid/validity checks, class-2 preservation |
# | Planet directories | independent site/date pairing | missing/ambiguous-pair tests and agreement |
# | Protocol | portable JSON | exact seed, membership, transforms and budgets |
# 

# ## Grading rubric (100 points)
# 
# | Assessed evidence | Points | Full-credit standard |
# |---|---:|---|
# | MNIST transforms and reusable loaders | 20 | Correct normalization; seeded 90/10 development split; untouched official test set; train-only augmentation; exact IDs returned |
# | EuroSAT100 loaders | 15 | Official 60/20/20 membership preserved; correct 13-band tensors; train-only shuffling; exact global IDs |
# | Planet catalog pipeline | 30 | Lazy paired reads; four image bands; labels 0/1/2 and ignore -1; CRS/transform/grid checks; image-valid normalization; paired augmentation |
# | Independent directory pairing | 15 | Site/date key matching; missing/ambiguous pairs rejected; site-aware development split; catalog agreement demonstrated |
# | Reproducibility protocol and evidence | 10 | Seed, all memberships, preprocessing, class mapping, budgets, shapes, figures, and rerun evidence saved |
# | Critical understanding answers | 10 | Specific, technically correct answers tied to observed evidence rather than generic definitions |
# | **Total** | **100** | |
# 
# Partial credit is based on demonstrated behavior, not the number of lines written.
# A loader that executes but leaks test data, misaligns masks, or discards class 2
# cannot receive full credit. Required evidence must remain visible in the submitted
# executed notebook.
# 

# ## Where you write code in the modular assignment
# 
# The notebooks are the **orchestrators**, not the main implementation files. Keep
# them open to read instructions, reload your modules, run checks, make figures, and
# write interpretations. Write assessed Python implementations in these exact files:
# 
# | Assignment | File you edit | Symbols you implement |
# |---|---|---|
# | A1 | `src/student_pipeline.py` | `paired_flip`, `normalize_image`, `training_transform`, `evaluation_transform`, `build_mnist_loaders`, `build_eurosat_loaders`, `PlanetCatalogDataset`, `match_directory_pairs`, `PlanetDirectoryDataset`, `build_planet_loaders`, `save_data_protocol`, `load_data_protocol` |
# | A2 | `src/student_models.py` | `MNISTResNet18`, `EuroSATClassifier`, `PlanetUNet` |
# | A2 | `src/student_training.py` | `run_epoch`, `fit` |
# | A3 | `src/student_inference.py` | `predict_classification`, `export_prediction`, `predict_and_export` |
# 
# Replace the `TODO`/`NotImplementedError` bodies while preserving the public
# function names, arguments, return contracts, and class attributes described in
# their docstrings. Save the module, rerun the notebook's reload cell, and recreate
# any datasets, loaders, models, or optimizers that were built from the old code.
# 
# You also write explanatory work in the notebook's clearly labelled evidence,
# architecture, interpretation, and critical-question Markdown cells, and document
# agent assistance in `AI_USE.md`. Do **not** implement assessed work in notebook
# scratch cells or edit `assignment_setup.py`, `student_checks.py`, `assessment_support.py`,
# the catalog, or the tests. Those supplied files define setup, checks, and evidence
# contracts and should remain unchanged.
# 

# ## Foundations: what the data pipeline is responsible for
# 
# A PyTorch `Dataset` defines the meaning of one indexed sample. A `DataLoader`
# controls how those samples become batches: ordering, shuffling, collation, worker
# processes, and batch size. Keeping these responsibilities separate is important.
# If a dataset silently chooses a different label or transform each time it is
# constructed, later assignments cannot reproduce A1 even if their loader settings
# look identical.
# 
# Three splits serve different purposes:
# 
# - **training** updates parameters and may use stochastic augmentation;
# - **validation** supports development decisions and checkpoint selection;
# - **test** is opened only after the full procedure is frozen.
# 
# A seed makes a randomized operation repeatable only when the same algorithm,
# input membership, and library behavior are used. Therefore this assignment also
# saves explicit IDs. The IDs—not the seed alone—are the durable handoff to A2/A3.
# 
# MNIST is the controlled introduction: one grayscale channel, fixed 28×28 pixels,
# ten classes, and a packaged download. EuroSAT100 introduces 13 spectral bands and
# an existing official split. Planet introduces the harder EO case: imagery and
# categorical masks are separate georeferenced files, invalid pixels exist, and a
# spatial transform must keep every image pixel aligned with its target pixel.
# 
# ### Shapes to reason about before coding
# 
# | Dataset | One image | One target | One batch |
# |---|---|---|---|
# | MNIST | `[1,28,28]` float | scalar class 0–9 | `[B,1,28,28]`, `[B]` |
# | EuroSAT100 | `[13,64,64]` float | scalar class 0–9 | `[B,13,64,64]`, `[B]` |
# | Planet | `[4,H,W]` float | `[H,W]` class map | `[B,4,H,W]`, `[B,H,W]` |
# 
# Classification assigns one label to an image. Segmentation assigns one label to
# each valid pixel. Do not add a singleton channel to an integer segmentation target
# for cross entropy, and do not one-hot encode it unless an explicitly chosen loss
# requires that representation.
# 

# ## Setup: upload the complete folder
# 
# **2i2c is the primary platform.** Upload or extract the entire folder as
# `/home/jovyan/assignments`, then open this notebook from the folder root. Keep
# `src/`, `catalog.csv`, all notebooks, and `outputs/` together.
# The `/home/jovyan` directory is persistent across normal 2i2c sessions.
# 
# The setup below adds `PROJECT_ROOT/src` directly to Python's import path. It does
# not create an environment or install the assignment as a package. It first checks
# the modules already supplied by the 2i2c image. Only if that check reports a
# missing package should you open a terminal in `assignments` and run:
# 
# ```bash
# python -m pip install -r requirements.txt
# ```
# 
# **Colab fallback:** put the complete folder in
# `My Drive/IDLEO/assignments`, open the notebook in Colab, and run the same
# cells. Code, protocols, outputs, and checkpoints remain in Drive. Runtime data
# may need to be downloaded again after a reset.
# 

# In[3]:


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


# ## Agentic coding without outsourcing the assignment
# 
# An agent may help you plan, inspect an error, propose a test, or review one saved
# function. Work on one named TODO at a time. Give the function contract, shapes,
# and the failing check; ask for an explanation of the proposed change. Read every
# line, reject unsupported assumptions, save the file yourself, reload it, and run
# both the supplied check and an independent counterexample. Do not ask an agent to
# "complete the assignment" or paste an answer repository. Record what you accepted,
# changed, rejected, and verified in `AI_USE.md`. You must be able to explain and
# modify the final code without assistance.
# 

# ## How to complete source-file tasks
# 
# Open `src/student_pipeline.py` in the JupyterLab file browser. Read each
# complete docstring, replace only its `NotImplementedError`, and save.
# Rerun the reload cell after every edit. Existing dataset objects retain old
# class definitions, so recreate them after reloading.
# 

# In[ ]:


import matplotlib.pyplot as plt
import torch

import student_checks as checks
import student_pipeline as pipeline
from assessment_support import (
    fetch_planet_pairs,
    set_seed,
    show_eurosat_batch,
    show_planet_triplet,
)

pipeline = importlib.reload(pipeline)
checks = importlib.reload(checks)
set_seed(42)


# ## Choose the data budget before inspecting outcomes
# 
# These small values make setup and debugging possible on CPU. Increase them
# only under the course assessment instructions. Record every change before
# looking at validation or test results.
# 

# In[ ]:


SEED = 42
MNIST_LIMITS = {"train": 512, "validate": 128, "test": 128}
EUROSAT_LIMITS = {"train": 60, "validate": 20, "test": 20}
PLANET_LIMITS = {"train": 8, "validate": 4, "test": 4}
MNIST_BATCH_SIZE = 64
EURO_BATCH_SIZE = 16
PLANET_BATCH_SIZE = 2
DOWNLOAD_MISSING_PLANET = True


# ## Task 1: MNIST transforms, split, loaders, and visualization (20 points)
# 
# Implement `build_mnist_loaders` before working with EO rasters. Start from
# the packaged 60,000-image training pool and official 10,000-image test pool.
# Use the seed to create a 90/10 training/validation split, but save the exact
# selected global IDs. Training may rotate digits; validation/test may not.
# Normalize all splits with mean `0.1307` and standard deviation `0.3081`.
# 
# **Ordered implementation:** (1) define separate train/evaluation transforms;
# (2) construct the packaged datasets; (3) generate or replay membership;
# (4) create three loaders with only training shuffled; (5) return loaders and
# IDs; (6) verify shapes, normalized values, labels, disjointness, and repeated
# evaluation batches. The limits below are a runtime budget, not a new split.
# 

# In[ ]:


mnist = pipeline.build_mnist_loaders(
    paths.data_root / "mnist",
    batch_size=MNIST_BATCH_SIZE,
    seed=SEED,
    limits=MNIST_LIMITS,
    download=True,
)
mnist_images, mnist_labels = next(iter(mnist["train"]))
print("MNIST:", mnist_images.shape, mnist_labels.shape)
print(
    "membership sizes:",
    {key: len(value) for key, value in mnist["split_ids"].items()},
)
assert mnist_images.shape[1:] == (1, 28, 28)
assert set(mnist["split_ids"]["train"]).isdisjoint(
    mnist["split_ids"]["validate"]
)
figure, axes = plt.subplots(2, 4, figsize=(8, 4))
for axis, image, label in zip(axes.flat, mnist_images[:8], mnist_labels[:8]):
    axis.imshow(image[0].cpu(), cmap="gray")
    axis.set_title(f"label {int(label)}")
    axis.axis("off")
figure.tight_layout()
figure.savefig(paths.output_root / "assignment_1/mnist_batch.png", dpi=150)


# **Evidence to submit (20 points):** implementation, transform explanation,
# membership counts and overlap assertions, tensor/label shapes, eight-image
# figure, and a short interpretation of values after normalization. Explain
# why the official test pool is not split or inspected during development.
# 

# ## Task 2: EuroSAT100 loaders (15 points)
# 
# Implement `build_eurosat_loaders`. Preserve EuroSAT100's documented
# 60/20/20 split and return its exact global integer membership. Shuffle only training. Explain the
# `[B,13,H,W]` image and `[B]` label shapes and why validation/test membership
# must not be regenerated in later assignments.
# 

# In[ ]:


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
euro_figure.savefig(
    paths.output_root / "assignment_1/eurosat_batch.png", dpi=150
)


# **Written evidence:** describe the split, preprocessing, channel meanings,
# tensor shapes, and one check showing train/validate/test IDs are disjoint.
# 

# ## Task 3: Planet catalog dataset and loaders (30 points)
# 
# Implement paired transforms, image-only normalization,
# `PlanetCatalogDataset`, and `build_planet_loaders`. Planet masks retain
# `0=noncrop`, `1=field interior`, `2=boundary`; invalid targets are `-1`.
# Verify image/mask CRS, affine transform, and grid before accepting a pair.
# Training may use paired stochastic flips; validation and test must remain
# deterministic. Never use test labels to choose preprocessing or models.
# 

# In[ ]:


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
planet_figure.savefig(
    paths.output_root / "assignment_1/planet_pair.png", dpi=150
)
checks.check_transforms(pipeline)
checks.check_loaders(eurosat, planet, mnist)


# **Independent verification:** add at least two counterexamples, such as an
# asymmetric marker proving paired flips, a constant band with invalid pixels,
# or a raster pair with mismatched transform. A loader that runs can still be
# scientifically wrong.
# 

# ## Task 4: Independent directory pairing (15 points)
# 
# Implement `match_directory_pairs` and `PlanetDirectoryDataset`. Match image
# and mask by site and acquisition date, not equal filename stems. Use only
# selected train+validate image names, reject missing/ambiguous masks, keep all
# dates from one site in the same development split, and compare one
# unaugmented directory sample with the catalog dataset.
# 

# In[ ]:


from pathlib import Path

development_rows = selected_catalog[
    selected_catalog.split.isin(["train", "validate"])
]
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
train_pairs = [
    pair for pair in pairs if pair[0].stem.split("_")[0] in train_sites
]
validation_pairs = [
    pair for pair in pairs if pair[0].stem.split("_")[0] not in train_sites
]
directory_train = pipeline.PlanetDirectoryDataset(
    train_pairs, pipeline.training_transform
)
directory_validate = pipeline.PlanetDirectoryDataset(
    validation_pairs, pipeline.evaluation_transform
)
print(
    "directory train/validate:", len(directory_train), len(directory_validate)
)


# ## Task 5: Save the protocol for A2 and A3 (10 points)
# 
# This file is the handoff between assignments. It must contain relative or
# integer identifiers—not machine-specific absolute paths.
# 

# In[ ]:


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


# ## Critical understanding questions (10 points)
# 
# Answer in your own words and refer to evidence from this notebook.
# 
# 1. How could sharing one mutable transform object between training and validation cause leakage or nondeterminism?
# 2. Why is replaying explicit MNIST IDs stronger evidence than merely reusing seed 42?
# 3. Why must the same spatial augmentation be applied to imagery and categorical masks?
# 4. Why does matching image and mask array shape not prove they describe the same geographic grid?
# 5. Why must missing reference labels not change image normalization or inference validity?
# 6. What information can per-image normalization remove from multispectral imagery?
# 7. Why do non-overlapping catalog IDs not prove geographic independence?
# 8. Which exact artifact ensures A2 and A3 use the same examples as A1, and how would you verify it?
# 9. What evidence would reveal that class `2` boundaries had accidentally been converted to binary foreground?
# 10. Contrast one-image classification targets with per-pixel segmentation targets and their batch shapes.
# 

# 1. It's my understanding that sharing one mutable transform object between training and validation can cause leakage or nondeterminism because the point of validation is to test the model on data it has never seen. Mutable transform objects can be changed (as opposed to immutable objects which are fixed), which means that training or validation data can affect each other if one is shared between.
# 

# In[ ]:




