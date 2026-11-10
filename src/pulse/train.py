"""Portable two-stage training recipe with validation-only model selection."""

import hashlib
import json
import random
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from .data import DATASETS, normalize, read_image, read_mask
from .evaluate import evaluate
from .geometry import pair_geometry
from .inference import decode_video, load_weights, sha256
from .metrics import summarize_contours, summarize_lvef
from .models.core.sam2_utils import load_pretrained_weights
from .sampling import refinement_clip, tiled_clips


def _seed(video_id, epoch, seed):
    return (
        seed
        + epoch * 100003
        + int(hashlib.sha256(video_id.encode()).hexdigest()[:8], 16)
    )


def _items(videos, dataset, stage, epoch, seed):
    items = []
    for video in videos:
        if stage == 2:
            ids = refinement_clip(
                len(video.frames), *video.endpoints, _seed(video.video_id, epoch, seed)
            )
            items.append((video, ids))
        else:
            interval = (
                range(len(video.frames))
                if dataset == "private-a4c"
                else video.interval()
            )
            items.extend(
                (video, ids)
                for ids in tiled_clips(interval)
                if any(i in video.masks for i in ids)
            )
    return items


def _load_clip(video, indices, spec, seed):
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as transforms

    rng = random.Random(seed)
    angle = rng.uniform(-10, 10)
    shift = [round(rng.uniform(-0.05, 0.05) * spec.size) for _ in range(2)]
    brightness, contrast = rng.uniform(0.8, 1.2), rng.uniform(0.8, 1.2)
    images, labels, positions = [], [], []
    for position, index in enumerate(indices):
        image = read_image(video.frames[index], spec.size)
        image = transforms.affine(
            image, angle, shift, 1.0, [0, 0], InterpolationMode.BILINEAR
        )
        image = transforms.adjust_contrast(
            transforms.adjust_brightness(image, brightness), contrast
        )
        images.append(normalize(image))
        if index in video.masks:
            mask = Image.fromarray(read_mask(video.masks[index], spec))
            mask = transforms.affine(
                mask, angle, shift, 1.0, [0, 0], InterpolationMode.NEAREST
            )
            labels.append(torch.from_numpy(np.array(mask, dtype=np.int64)))
            positions.append(position)
    return torch.stack(images), torch.stack(labels), positions


def _signed_distance(binary):
    import cv2

    if not binary.any():
        return np.zeros_like(binary, dtype=np.float32)
    inside = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    outside = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 3)
    return outside - (inside - 1) * binary


def segmentation_loss(logits, labels, boundary_weight):
    probabilities = logits.float().softmax(1)[:, 1:]
    target = (
        F.one_hot(labels, num_classes=logits.shape[1])
        .permute(0, 3, 1, 2)[:, 1:]
        .float()
    )
    eps = 1e-6
    overlap = (probabilities * target).sum((2, 3))
    denominator = (probabilities.square() + target.square()).sum((2, 3))
    dice_loss = (1 - (2 * overlap + eps) / (denominator + eps)).sum(1).mean()
    loss = 0.75 * F.cross_entropy(logits.float(), labels) + 2.0 * dice_loss
    if boundary_weight:
        masks = target.detach().cpu().numpy().astype(np.uint8)
        distances = np.stack(
            [np.stack([_signed_distance(mask) for mask in item]) for item in masks]
        )
        distance_tensor = torch.from_numpy(distances).to(logits.device)
        loss += boundary_weight * (probabilities * distance_tensor).mean()
    return loss


def _save(path, state):
    pending = path.with_suffix(".tmp")
    torch.save(state, pending)
    pending.replace(path)


def train(model, train_videos, val_videos, args, device, precision):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.stage == 2:
        if args.dataset != "echonet" or any(
            v.lvef is None for v in train_videos + val_videos
        ):
            raise ValueError(
                "Stage 2 requires EchoNet and clinical LVEF for train and validation"
            )
        initialization_hash = load_weights(model, args.initial)
    elif args.initial:
        initialization_hash = sha256(args.initial)
        load_pretrained_weights(model, str(args.initial), strict=False)
        coverage = model.initialization_audit["module_coverage"]
        for prefix in ("image_encoder.", "memory_attention."):
            if coverage[prefix]["loaded_keys"] != coverage[prefix]["keys"]:
                raise ValueError(f"Generic initialization did not cover {prefix}")
    else:
        raise ValueError("Provide a generic SAM2 Hiera Base+ checkpoint with --initial")

    model.to(device)
    spec = DATASETS[args.dataset]
    encoder = list(model.image_encoder.parameters())
    encoder_ids = {id(p) for p in encoder}
    other = [p for p in model.parameters() if id(p) not in encoder_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder, "lr": 1e-5},
            {"params": other, "lr": 1e-4},
        ],
        weight_decay=1e-4,
    )
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=19,
        threshold=0.02 if args.stage == 2 else 1e-5,
        threshold_mode="abs",
        min_lr=1e-6,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    if args.stage == 1:
        (args.output / "initialization.json").write_text(
            json.dumps(model.initialization_audit, indent=2), encoding="utf-8"
        )
    (args.output / "recipe.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "stage": args.stage,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "precision": precision,
                "memory": args.memory,
                "initial_sha256": initialization_hash,
                "seed": args.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    best_dice, best_mae, bad_epochs = -1.0, float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        items = _items(train_videos, args.dataset, args.stage, epoch, args.seed)
        if not items:
            raise ValueError("The training manifest produced no labeled clips")
        random.Random(args.seed + epoch).shuffle(items)
        buckets = defaultdict(list)
        for item in items:
            buckets[len(item[1])].append(item)
        batches = [
            bucket[start : start + args.batch_size]
            for bucket in buckets.values()
            for start in range(0, len(bucket), args.batch_size)
        ]
        random.Random(args.seed + epoch).shuffle(batches)
        running_loss = 0.0
        for step, batch in enumerate(batches):
            if args.stage == 1 and epoch <= 8:
                fraction = ((epoch - 1) * len(batches) + step) / (8 * len(batches))
                for group, base_lr in zip(optimizer.param_groups, (1e-5, 1e-4)):
                    group["lr"] = base_lr * (0.001 + 0.999 * fraction)
            clips = [
                _load_clip(video, ids, spec, _seed(video.video_id, epoch, args.seed))
                for video, ids in batch
            ]
            images = torch.stack([clip[0] for clip in clips])
            logits = torch.stack(
                list(decode_video(model, [images], device, precision)), dim=1
            )
            annotated = torch.cat([logits[i, clip[2]] for i, clip in enumerate(clips)])
            targets = torch.cat([clip[1] for clip in clips]).to(device)
            weight = 0.2 if args.stage == 2 else 0.2 * min(epoch / 60, 1.0)
            loss = segmentation_loss(annotated, targets, weight)
            if args.stage == 2:
                endpoints = torch.stack(
                    [
                        logits[i, [ids.index(video.ed), ids.index(video.es)]]
                        for i, (video, ids) in enumerate(batch)
                    ]
                )
                prediction = (
                    pair_geometry(endpoints.float().softmax(2)[:, :, 1])["raw_ef"] / 100
                )
                reference = prediction.new_tensor([v.lvef / 100 for v, _ in batch])
                loss += (prediction - reference).abs().mean()
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss at epoch {epoch}, step {step}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            running_loss += float(loss.detach())
        model.eval()
        spatial, functional = evaluate(
            model,
            val_videos,
            args.dataset,
            device,
            precision,
            batch_size=4,
            chunk_budget=32 if args.dataset == "echonet" else 8,
            calculate_lvef=args.stage == 2,
            refinement_seed=20261024 if args.stage == 2 else None,
        )
        unit = "patient_id" if args.dataset == "camus" else "video_id"
        dice = summarize_contours(spatial, unit)["all"]["dice"]["mean"]
        ef = summarize_lvef(functional).get("metrics") if functional else None
        mae = ef["MAE"] if ef else None
        state = {
            "model": model.state_dict(),
            "epoch": epoch,
            "validation_dice": dice,
            "validation_mae": mae,
        }
        if dice > best_dice:
            best_dice = dice
            _save(args.output / "best_dice.pt", state)
        if args.stage == 2:
            if dice >= 0.92 and mae is not None and mae < best_mae:
                best_mae, bad_epochs = mae, 0
                _save(args.output / "best_lvef.pt", state)
            else:
                bad_epochs += 1
            plateau.step(mae if dice >= 0.92 and mae is not None else float("inf"))
        elif epoch > 8:
            plateau.step(-dice)
        with (args.output / "history.jsonl").open("a", encoding="utf-8") as history:
            history.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_loss": running_loss / len(batches),
                        "validation_dice": dice,
                        "validation_mae": mae,
                    }
                )
                + "\n"
            )
        _save(args.output / "last.pt", state)
        if args.stage == 2 and bad_epochs >= 50:
            break
