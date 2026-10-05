"""Assignment 2 training loops, reused unchanged for Assignment 3 evaluation."""


def run_epoch(model, loader, device, task, optimizer=None):
    """Run one complete classification or three-class segmentation epoch.

    With an optimizer, train every batch; otherwise evaluate without gradients.
    Move data to the device, clear gradients, and compute raw logits and cross
    entropy. Backpropagate only during training. Aggregate loss per example,
    rather than by averaging batch means. Segmentation uses ignore_index=-1.
    Return global
    confusion-derived metrics from all batches, including per-class results.
    """
    raise NotImplementedError("A2: implement run_epoch")


def fit(
    model,
    train_loader,
    validation_loader,
    device,
    task,
    optimizer,
    epochs,
    checkpoint_path,
    model_config,
):
    """Train, validate, and save only the minimum-validation-loss checkpoint.

    The checkpoint must contain model_state, model_config, epoch, task, and
    validation_loss. Return a pandas DataFrame with one row per epoch. Never use
    the test loader for selection.
    """
    raise NotImplementedError("A2: implement fit")
