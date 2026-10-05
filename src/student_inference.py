"""Assignment 3 inference and georeferenced export implementations."""


def predict_classification(model, loader, device):
    """Run held-out EuroSAT batches in eval/no-grad mode and return metrics."""
    raise NotImplementedError("A3: implement held-out EuroSAT inference")


def export_prediction(prediction, image_valid, source_path, destination):
    """Write one source-grid uint8 GeoTIFF with classes 0/1/2 and nodata 255."""
    raise NotImplementedError("A3: implement georeferenced Planet export")


def predict_and_export(model, loader, device, output_dir):
    """Run all Planet test batches, aggregate metrics, and export maps.

    Return global three-class metrics plus one record per chip containing its
    source/output paths, output hash, and per-chip metrics for a predetermined
    failure-analysis ranking.
    """
    raise NotImplementedError("A3: implement Planet inference and export")
