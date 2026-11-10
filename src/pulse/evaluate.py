"""Evaluate original contours and, on EchoNet, fixed annotated ED/ES LVEF."""

import csv
import json
from collections import defaultdict

import numpy as np
import torch

from .data import DATASETS, normalize, read_image, read_mask
from .geometry import pair_geometry
from .inference import decode_legacy, decode_video
from .metrics import contour_metrics, summarize_contours, summarize_lvef
from .sampling import refinement_clip


def _chunks(videos, indices, size, frames_per_chunk):
    for start in range(0, len(indices), frames_per_chunk):
        part = indices[start : start + frames_per_chunk]
        yield torch.stack(
            [
                torch.stack(
                    [normalize(read_image(video.frames[i], size)) for i in part]
                )
                for video in videos
            ]
        )


def _contour_rows(video, predictions, dataset, spec):
    rows = []
    for index, mask in sorted(video.masks.items()):
        if dataset != "private-a4c" and index not in video.endpoints:
            continue
        target = read_mask(mask, spec)
        for label_id, label in enumerate(spec.classes, start=1):
            pred = predictions[index] == label_id
            ref = target == label_id
            policy = "diagonal" if dataset == "private-a4c" else "undefined"
            dice, hd95, asd, status = contour_metrics(pred, ref, video.spacing, policy)
            phase = "ED" if index == video.ed else "ES" if index == video.es else ""
            rows.append(
                {
                    "video_id": video.video_id,
                    "patient_id": video.patient_id,
                    "view": video.view,
                    "frame": index,
                    "phase": phase,
                    "class": label,
                    "dice": dice,
                    "hd95": hd95,
                    "asd": asd,
                    "contour_status": status,
                }
            )
    return rows


def _function_rows(videos, predictions, spec, device, use_gt):
    if any(video.lvef is None for video in videos):
        raise ValueError("Every EchoNet case must have a clinical LVEF reference")
    pairs = np.stack(
        [
            np.stack(
                [
                    read_mask(video.masks[i], spec) if use_gt else prediction[i]
                    for i in video.endpoints
                ]
            )
            for video, prediction in zip(videos, predictions)
        ]
    )
    geometry = pair_geometry(torch.from_numpy(pairs == 1).to(device).float())
    rows = []
    for i, video in enumerate(videos):
        ef = float(geometry["raw_ef"][i]) if bool(geometry["valid"][i]) else None
        ed_volume, es_volume = geometry["volume"][i].tolist()
        rows.append(
            {
                "video_id": video.video_id,
                "ed": video.ed,
                "es": video.es,
                "reference": video.lvef,
                "prediction": ef,
                "ed_volume_px3": ed_volume,
                "es_volume_px3": es_volume,
                "absolute_error": abs(ef - video.lvef) if ef is not None else None,
            }
        )
    return rows


def _check_logits(logits):
    if not torch.isfinite(logits).all():
        raise RuntimeError("Non-finite model output")
    return logits.argmax(1).to(device="cpu", dtype=torch.uint8).numpy()


@torch.inference_mode()
def evaluate(
    model,
    videos,
    dataset,
    device,
    precision,
    batch_size,
    chunk_budget,
    use_gt=False,
    calculate_lvef=True,
    refinement_seed=None,
):
    spec = DATASETS[dataset]
    spatial, functional = [], []
    if dataset == "echonet":
        buckets = defaultdict(list)
        for video in videos:
            indices = video.interval()
            if refinement_seed is not None:
                indices = refinement_clip(
                    len(video.frames), *video.endpoints, refinement_seed
                )
            buckets[len(indices)].append((video, indices))
        for length, bucket in sorted(buckets.items()):
            for start in range(0, len(bucket), batch_size):
                entries = bucket[start : start + batch_size]
                group = [video for video, _ in entries]
                if use_gt:
                    functional.extend(
                        _function_rows(group, [None] * len(group), spec, device, True)
                    )
                    continue
                # Offsets align chronological frames across cases without equalizing ED/ES order.
                chunk = max(1, chunk_budget // len(group))
                inputs = (
                    torch.stack(
                        [
                            torch.stack(
                                [
                                    normalize(read_image(video.frames[i], spec.size))
                                    for i in indices[left : left + chunk]
                                ]
                            )
                            for video, indices in entries
                        ]
                    )
                    for left in range(0, length, chunk)
                )
                endpoints = [dict() for _ in group]
                for offset, logits in enumerate(
                    decode_video(model, inputs, device, precision)
                ):
                    masks = _check_logits(logits)
                    for j, (video, indices) in enumerate(entries):
                        frame = indices[offset]
                        if frame in video.endpoints:
                            endpoints[j][frame] = masks[j]
                for video, predicted in zip(group, endpoints):
                    spatial.extend(_contour_rows(video, predicted, dataset, spec))
                if calculate_lvef:
                    functional.extend(
                        _function_rows(group, endpoints, spec, device, False)
                    )
    else:
        for video in videos:
            ids = range(len(video.frames))
            if dataset == "camus":
                ids = video.interval()
            inputs = _chunks([video], ids, spec.size, chunk_budget)
            if model.pulse_profile == "core":
                outputs = decode_video(model, inputs, device, precision)
            else:
                outputs = decode_legacy(
                    model, (chunk[0] for chunk in inputs), device, precision
                )
            predictions = {}
            for frame, logits in zip(ids, outputs, strict=True):
                if frame in video.masks:
                    predictions[frame] = _check_logits(logits)[0]
            spatial.extend(_contour_rows(video, predictions, dataset, spec))
    return spatial, functional


def write_evaluation(output, dataset, videos, spatial, functional, metadata):
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in [("contours", spatial), ("lvef", functional)]:
        if rows:
            with (output / f"{name}.csv").open(
                "w", encoding="utf-8", newline=""
            ) as file:
                writer = csv.DictWriter(file, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
    result = metadata | {
        "videos": len(videos),
        "patients": len({video.patient_id for video in videos}),
        "dataset": dataset,
        "distance_unit": DATASETS[dataset].distance_unit,
        "contours": summarize_contours(
            spatial, "patient_id" if dataset == "camus" else "video_id"
        ),
        "one_empty_contours": sum(
            row["contour_status"] == "one_empty" for row in spatial
        ),
        "lvef": summarize_lvef(functional) if functional else None,
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result
