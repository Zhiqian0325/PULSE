"""Explicit manifest schema and image loading. Indices are zero-based."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class DatasetSpec:
    size: int
    classes: tuple[str, ...]
    distance_unit: str


DATASETS = {
    "echonet": DatasetSpec(128, ("LV",), "px"),
    "camus": DatasetSpec(256, ("LV", "MYO", "LA"), "mm"),
    "private-a4c": DatasetSpec(256, ("LV", "RV", "RA", "LA"), "px"),
}


@dataclass(frozen=True)
class Video:
    video_id: str
    patient_id: str
    split: str
    frames: tuple[Path, ...]
    masks: dict[int, Path]
    ed: int | None
    es: int | None
    lvef: float | None
    spacing: tuple[float, float]
    view: str | None

    @property
    def endpoints(self):
        if self.ed is None or self.es is None:
            raise ValueError(f"{self.video_id}: ED and ES are required")
        return self.ed, self.es

    def interval(self):
        first, last = sorted(self.endpoints)
        return range(first, last + 1)


def relative_file(root: Path, path: str) -> Path:
    relative = Path(path)
    if relative.anchor or ".." in relative.parts or "\\" in path or ":" in path:
        raise ValueError(f"Expected a portable path relative to the data root: {path}")
    file = (root / relative).resolve()
    if not file.is_relative_to(root) or not file.is_file():
        raise ValueError(f"Missing file or path outside the data root: {path}")
    return file


def load_manifest(path: Path, root: Path, dataset: str, split: str):
    root = root.resolve(strict=True)
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if raw.get("schema_version") != 1 or raw.get("dataset") != dataset:
        raise ValueError("Manifest schema or dataset does not match the request")
    seen = set()
    patient_splits = {}
    selected = []
    for row in raw["videos"]:
        video_id, patient_id, fold = row["video_id"], row["patient_id"], row["split"]
        if (
            not video_id
            or not patient_id
            or video_id in seen
            or fold not in {"train", "val", "test"}
        ):
            raise ValueError(f"Invalid or duplicate video entry: {video_id}")
        seen.add(video_id)
        if patient_splits.setdefault(patient_id, fold) != fold:
            raise ValueError(f"Patient {patient_id} appears in multiple splits")
        # Check all split identities without opening files from other partitions.
        if fold != split:
            continue
        frames = tuple(relative_file(root, file) for file in row["frames"])
        masks = {
            int(index): relative_file(root, file)
            for index, file in row["masks"].items()
        }
        if len(set(frames)) != len(frames) or not frames or not masks:
            raise ValueError(f"{video_id}: missing frames/masks or duplicate frames")
        if len(masks) != len(row["masks"]) or any(
            not 0 <= i < len(frames) for i in masks
        ):
            raise ValueError(f"{video_id}: invalid mask frame indices")
        ed, es = row.get("ed"), row.get("es")
        if dataset != "private-a4c" or ed is not None or es is not None:
            if (
                type(ed) is not int
                or type(es) is not int
                or ed == es
                or not {ed, es} <= masks.keys()
            ):
                raise ValueError(
                    f"{video_id}: ED and ES must be distinct annotated frame indices"
                )
        lvef = row.get("lvef_percent")
        if lvef is not None:
            lvef = float(lvef)
            if not 0 <= lvef <= 100:
                raise ValueError(f"{video_id}: invalid clinical LVEF percentage")
        spacing = (1.0, 1.0)
        if dataset == "camus":
            spacing = tuple(map(float, row["spacing_yx_mm"]))
            if len(spacing) != 2 or any(
                not math.isfinite(x) or x <= 0 for x in spacing
            ):
                raise ValueError(f"{video_id}: invalid CAMUS evaluation spacing")
            if row.get("view") not in {"2CH", "4CH"}:
                raise ValueError(f"{video_id}: CAMUS view must be 2CH or 4CH")
        selected.append(
            Video(
                video_id,
                patient_id,
                fold,
                frames,
                masks,
                ed,
                es,
                lvef,
                spacing,
                row.get("view"),
            )
        )
    if not selected:
        raise ValueError(f"No {split} videos in the manifest")
    if dataset == "camus":
        views = {}
        for video in selected:
            views.setdefault(video.patient_id, []).append(video.view)
        if any(sorted(pair) != ["2CH", "4CH"] for pair in views.values()):
            raise ValueError("Each CAMUS patient must have one 2CH and one 4CH video")
    return selected


def read_image(path: Path, size: int):
    with Image.open(path) as image:
        return image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)


def normalize(image: Image.Image) -> torch.Tensor:
    rgb = (
        torch.from_numpy(np.array(image, dtype=np.uint8)).permute(2, 0, 1).float() / 255
    )
    mean = rgb.new_tensor([0.485, 0.456, 0.406])[:, None, None]
    std = rgb.new_tensor([0.229, 0.224, 0.225])[:, None, None]
    return (rgb - mean) / std


def read_mask(path: Path, spec: DatasetSpec) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode not in {"P", "L"}:
            raise ValueError(f"{path.name}: expected an indexed class mask (P or L)")
        mask = np.asarray(
            image.resize((spec.size, spec.size), Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
    if mask.max() > len(spec.classes):
        raise ValueError(f"{path.name}: label outside 0..{len(spec.classes)}")
    return mask
