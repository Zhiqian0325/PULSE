"""Convert an explicit cohort CSV and prepared JPEG/PNG directories to a manifest."""

import argparse
import csv
import json
from pathlib import Path


def _directory(root, name):
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_dir():
        raise ValueError(f"Expected a directory below the data root: {name}")
    return path


def _indexed(directory, suffix):
    indexed = {}
    for file in directory.iterdir():
        if file.suffix.lower() == suffix and file.is_file():
            if not file.stem.isascii() or not file.stem.isdecimal():
                raise ValueError(f"Expected a numeric frame name: {file.name}")
            index = int(file.stem)
            if index in indexed:
                raise ValueError(f"Duplicate frame number: {file.name}")
            indexed[index] = file
    return indexed


def make_row(root, row, dataset):
    frames = _indexed(_directory(root, row["frames_dir"]), ".jpg")
    masks = _indexed(_directory(root, row["masks_dir"]), ".png")
    if sorted(frames) != list(range(len(frames))) or not frames or not masks:
        raise ValueError(
            f"{row['video_id']}: expected contiguous zero-based JPEGs and indexed PNG masks"
        )
    entry = {
        "video_id": row["video_id"],
        "patient_id": row["patient_id"],
        "split": row["split"],
        "frames": [frames[i].relative_to(root).as_posix() for i in range(len(frames))],
        "masks": {
            str(i): file.relative_to(root).as_posix()
            for i, file in sorted(masks.items())
        },
    }
    if dataset != "private-a4c":
        entry.update(ed=int(row["ed"]), es=int(row["es"]))
    if dataset == "echonet":
        entry["lvef_percent"] = float(row["lvef_percent"])
    if dataset == "camus":
        entry.update(
            view=row["view"],
            spacing_yx_mm=[float(row["spacing_y_mm"]), float(row["spacing_x_mm"])],
        )
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=["echonet", "camus", "private-a4c"], required=True
    )
    parser.add_argument("--cohort-csv", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root.resolve(strict=True)
    with args.cohort_csv.open(encoding="utf-8-sig", newline="") as file:
        videos = [make_row(root, row, args.dataset) for row in csv.DictReader(file)]
    if not videos:
        parser.error("Cohort CSV is empty")
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(
            {"schema_version": 1, "dataset": args.dataset, "videos": videos},
            file,
            indent=2,
        )
        file.write("\n")


if __name__ == "__main__":
    main()
