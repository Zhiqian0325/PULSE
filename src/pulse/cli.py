"""Command-line entry points for dataset audit, evaluation and training."""

import argparse
import hashlib
import json
from importlib.resources import files
from pathlib import Path

import torch

from .data import DATASETS, load_manifest, read_mask
from .evaluate import evaluate, write_evaluation
from .inference import load_weights, make_model, sha256


ECHONET_PAPER_COHORT = (
    "aa3e1ad380f8b71111ffac738286650784e073efb2ff96f4a6782c67d4656132"
)


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def registry():
    return json.loads(
        files("pulse").joinpath("paper_checkpoints.json").read_text(encoding="utf-8")
    )


def add_data_options(parser):
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)


def add_execution_options(parser):
    parser.add_argument(
        "--device", default="cpu", help="cpu, cuda, or a CUDA index such as cuda:0"
    )
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--batch-size", type=positive_int, default=4)
    parser.add_argument("--output", type=Path, required=True)


def _device(args):
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"} or (
        device.type == "cuda" and not torch.cuda.is_available()
    ):
        raise ValueError("Select an available CPU or CUDA device")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if args.precision == "bf16" and device.type != "cuda":
        raise ValueError("Use FP32 for CPU execution")
    return device


def _eval(args):
    device = _device(args)
    videos = load_manifest(args.manifest, args.data_root, args.dataset, args.split)
    if args.output.exists():
        raise ValueError(f"Output already exists: {args.output}")
    if args.gt_geometry and args.dataset != "echonet":
        raise ValueError("The GT geometry reference is defined for EchoNet")
    paper = registry().get(args.paper_checkpoint)
    if paper and (args.dataset != paper["dataset"] or args.split != "test"):
        raise ValueError(
            "Paper checkpoint must use its recorded dataset and test split"
        )
    if paper:
        if len(videos) != paper["reference"]["videos"]:
            raise ValueError("Test video count does not match the paper registry")
        if args.dataset == "echonet":
            cohort = "\n".join(sorted(video.video_id for video in videos)) + "\n"
            if hashlib.sha256(cohort.encode()).hexdigest() != ECHONET_PAPER_COHORT:
                raise ValueError("EchoNet test identities differ from the paper cohort")
        patients = paper["reference"].get("patients")
        if patients and len({v.patient_id for v in videos}) != patients:
            raise ValueError("Test patient count does not match the paper registry")
        if args.memory is not None and args.memory != paper["memory"]:
            raise ValueError("The memory setting is locked by the paper checkpoint")
    model, digest = None, None
    memory = paper["memory"] if paper else args.memory is not False
    if not args.gt_geometry:
        model = make_model(
            args.dataset, memory=memory, archived=bool(paper) or args.archived_profile
        )
        digest = load_weights(
            model,
            args.checkpoint,
            paper["sha256"] if paper else None,
            paper.get("epoch") if paper else None,
        )
        model.to(device).eval()
    chunk = args.encoder_chunk or (32 if args.dataset == "echonet" else 8)
    spatial, functional = evaluate(
        model,
        videos,
        args.dataset,
        device,
        args.precision,
        args.batch_size,
        chunk,
        args.gt_geometry,
    )
    result = write_evaluation(
        args.output,
        args.dataset,
        videos,
        spatial,
        functional,
        {
            "manifest_sha256": sha256(args.manifest),
            "checkpoint_sha256": digest,
            "paper_checkpoint": args.paper_checkpoint,
            "split": args.split,
            "memory": memory if model else None,
            "architecture_profile": model.pulse_profile if model else None,
            "precision": args.precision,
            "batch_size": args.batch_size,
            "encoder_chunk_budget": chunk,
            "gt_geometry": args.gt_geometry,
        },
    )
    print(json.dumps(result, indent=2))


def _audit(args):
    report = {}
    for split in args.split:
        videos = load_manifest(args.manifest, args.data_root, args.dataset, split)
        if args.check_masks:
            for video in videos:
                for path in video.masks.values():
                    read_mask(path, DATASETS[args.dataset])
        report[split] = {
            "videos": len(videos),
            "patients": len({v.patient_id for v in videos}),
            "annotated_frames": sum(len(v.masks) for v in videos),
        }
    print(json.dumps(report, indent=2))


def _train(args):
    from .train import train

    device = _device(args)
    training = load_manifest(args.manifest, args.data_root, args.dataset, "train")
    validation = load_manifest(args.manifest, args.data_root, args.dataset, "val")
    torch.manual_seed(args.seed)
    train(
        make_model(args.dataset, args.memory),
        training,
        validation,
        args,
        device,
        args.precision,
    )


def main():
    parser = argparse.ArgumentParser(prog="pulse", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser(
        "audit-data", help="check the manifest, patient isolation and file availability"
    )
    add_data_options(audit)
    audit.add_argument(
        "--split",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val", "test"],
    )
    audit.add_argument("--check-masks", action="store_true")
    audit.set_defaults(action=_audit)

    evaluation = commands.add_parser(
        "evaluate", help="score a checkpoint or run the EchoNet GT geometry reference"
    )
    add_data_options(evaluation)
    add_execution_options(evaluation)
    mode = evaluation.add_mutually_exclusive_group(required=True)
    mode.add_argument("--checkpoint", type=Path)
    mode.add_argument("--gt-geometry", action="store_true")
    evaluation.add_argument("--paper-checkpoint", choices=registry())
    evaluation.add_argument(
        "--archived-profile",
        action="store_true",
        help="load a legacy CAMUS/private checkpoint",
    )
    evaluation.add_argument(
        "--memory", action=argparse.BooleanOptionalAction, default=None
    )
    evaluation.add_argument("--split", choices=["val", "test"], default="test")
    evaluation.add_argument("--encoder-chunk", type=positive_int)
    evaluation.set_defaults(action=_eval)

    training = commands.add_parser(
        "train", help="train a new model with the portable recipe"
    )
    add_data_options(training)
    add_execution_options(training)
    training.add_argument("--initial", type=Path, required=True)
    training.add_argument("--stage", type=int, choices=[1, 2], required=True)
    training.add_argument(
        "--memory", action=argparse.BooleanOptionalAction, default=True
    )
    training.add_argument("--epochs", type=positive_int, default=200)
    training.add_argument("--seed", type=int, default=42)
    training.set_defaults(action=_train)

    args = parser.parse_args()
    if getattr(args, "gt_geometry", False) and args.paper_checkpoint:
        parser.error("--gt-geometry cannot be combined with --paper-checkpoint")
    try:
        args.action(args)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"pulse: {error}\n")


if __name__ == "__main__":
    main()
