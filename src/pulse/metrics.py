"""Segmentation and function metrics with explicit missing-result accounting."""

from collections import defaultdict
from statistics import fmean, stdev

import numpy as np
from scipy import ndimage


def contour_metrics(prediction, target, spacing, empty_policy="undefined"):
    pixels = int(prediction.sum()) + int(target.sum())
    dice = 2 * int((prediction & target).sum()) / pixels if pixels else 1.0
    if not pixels:
        return dice, 0.0, 0.0, "both_empty"
    if not prediction.any() or not target.any():
        penalty = None
        if empty_policy == "diagonal":
            penalty = float(np.linalg.norm((np.asarray(target.shape) - 1) * spacing))
        return dice, penalty, penalty, "one_empty"
    structure = ndimage.generate_binary_structure(2, 1)
    ps = prediction & ~ndimage.binary_erosion(prediction, structure, border_value=0)
    ts = target & ~ndimage.binary_erosion(target, structure, border_value=0)
    distances = np.concatenate(
        [
            ndimage.distance_transform_edt(~ts, sampling=spacing)[ps],
            ndimage.distance_transform_edt(~ps, sampling=spacing)[ts],
        ]
    )
    return (
        dice,
        float(np.quantile(distances, 0.95)),
        float(distances.mean()),
        "nonempty",
    )


def summary(values):
    valid = [v for v in values if v is not None]
    return {
        "N": len(values),
        "valid_N": len(valid),
        "mean": fmean(valid) if valid else None,
        "SD": stdev(valid) if len(valid) > 1 else None,
    }


def summarize_contours(rows, unit):
    result = {}
    for label in ("all", *sorted({r["class"] for r in rows})):
        group = defaultdict(list)
        for row in rows:
            if label == "all" or label == row["class"]:
                group[row[unit]].append(row)
        metrics = {}
        for key in ("dice", "hd95", "asd"):
            values = []
            for records in group.values():
                scores = [r[key] for r in records if r[key] is not None]
                values.append(fmean(scores) if scores else None)
            metrics[key] = summary(values)
        result[label] = metrics
    return result


def summarize_lvef(rows):
    valid = [row for row in rows if row["prediction"] is not None]
    counts = {"N": len(rows), "valid_N": len(valid), "failed_N": len(rows) - len(valid)}
    if not valid:
        return counts | {"metrics": None}
    ref = np.array([row["reference"] for row in valid])
    pred = np.array([row["prediction"] for row in valid])
    error = pred - ref
    ss_ref = np.square(ref - ref.mean()).sum()
    metrics = {
        "MAE": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt(np.square(error).mean())),
        "bias": float(error.mean()),
        "r": float(np.corrcoef(ref, pred)[0, 1])
        if ss_ref > 0 and np.ptp(pred) > 0
        else None,
        "R2": float(1 - np.square(error).sum() / ss_ref) if ss_ref > 0 else None,
    }
    key = "metrics" if len(valid) == len(rows) else "valid_subset_diagnostic"
    return counts | {key: metrics}
