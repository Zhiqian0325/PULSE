"""Checkpoint loading and causal decoding with memory carried across chunks."""

import hashlib
import importlib
from contextlib import nullcontext
from pathlib import Path

import torch

from .data import DATASETS


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return (
            hashlib.file_digest(file, "sha256").hexdigest()
            if hasattr(hashlib, "file_digest")
            else _sha256_stream(file)
        )


def _sha256_stream(file):
    digest = hashlib.sha256()
    for block in iter(lambda: file.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def make_model(dataset: str, memory: bool = True, archived: bool = False):
    if not memory and dataset != "echonet":
        raise ValueError(
            "The independently trained memory-free configuration is EchoNet-only"
        )
    profile = dataset.replace("-", "_") if archived and dataset != "echonet" else "core"
    module = importlib.import_module(f".models.{profile}.sam2", __package__)
    spec = DATASETS[dataset]
    model = module.Sam2Vio(
        num_classes=len(spec.classes), image_size=spec.size, num_maskmem=7
    )
    # The memory-free EchoNet checkpoints retain the unused parameter keys.
    if not memory:
        model.num_maskmem = 0
    model.pulse_profile = profile
    return model


def load_weights(model, path, expected_hash=None, expected_epoch=None):
    digest = sha256(path)
    if expected_hash and digest != expected_hash:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {path.name}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if expected_epoch is not None and checkpoint.get("epoch") != expected_epoch:
        raise ValueError("Checkpoint epoch differs from the paper registry")
    state = checkpoint
    for key in ("model", "state_dict", "model_state_dict"):
        if key in checkpoint:
            state = checkpoint[key]
            break
    model.load_state_dict(state, strict=True)
    return digest


def autocast(device, precision):
    if precision == "fp32":
        return nullcontext()
    if device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise ValueError(
            "BF16 requires a CUDA device with BF16 support; use --precision fp32"
        )
    return torch.autocast("cuda", dtype=torch.bfloat16)


def decode_video(model, chunks, device, precision):
    """Yield [B,C,H,W] logits in order for successive [B,T,3,H,W] chunks."""
    history = []
    step = 0
    for images in chunks:
        batch, length = images.shape[:2]
        with autocast(device, precision):
            encoded = model.forward_image(images.flatten(0, 1).to(device))
            _, features, positions, sizes = model._prepare_backbone_features(encoded)
        for offset in range(length):
            indices = torch.arange(batch, device=device) * length + offset
            with autocast(device, precision):
                out = model.track_step(
                    step,
                    [f[:, indices] for f in features],
                    [p[:, indices] for p in positions],
                    sizes,
                    history,
                    run_mem_encoder=model.num_maskmem > 0,
                    num_frames=8,
                )
            if model.num_maskmem:
                history.append(
                    {
                        "memory_step": step,
                        "maskmem_features": out["maskmem_features"],
                        "maskmem_pos_enc": [out["maskmem_pos_enc"][-1]],
                        "obj_ptr": out["obj_ptr"],
                    }
                )
                history[:] = history[-model.num_maskmem :]
            yield out["pred_masks_high_res"]
            step += 1


def decode_legacy(model, chunks, device, precision):
    """Retain the archived CAMUS and private model's native pointer semantics."""
    from .models.common.legacy_utils import CircularList

    history = CircularList(model.num_maskmem)
    step = 0
    for images in chunks:
        with autocast(device, precision):
            encoded = model.forward_image(images.to(device))
            _, features, positions, sizes = model._prepare_backbone_features(encoded)
            outputs = []
            for offset in range(len(images)):
                index = torch.tensor([offset], device=device)
                frame = (
                    torch.tensor([step], device=device)
                    if model.pulse_profile == "camus"
                    else step
                )
                out = model.track_step(
                    frame,
                    [f[:, index] for f in features],
                    [p[:, index] for p in positions],
                    sizes,
                    history,
                )
                history.append(out)
                outputs.append(out["pred_masks_high_res"])
                step += 1
        yield from outputs
