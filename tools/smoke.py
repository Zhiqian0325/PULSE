"""Exercise the installed pipeline on generated, nonclinical images."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from PIL import Image

from pulse.data import load_manifest
from pulse.evaluate import evaluate
from pulse.inference import decode_video, make_model
from pulse.metrics import summarize_lvef


def synthetic_fixture(root):
    frames, masks = [], {}
    yy, xx = np.ogrid[:128, :128]
    for index in range(10):
        radius = 27 - index
        cavity = ((yy - 63) / (radius + 10)) ** 2 + ((xx - 63) / radius) ** 2 < 1
        image = Image.fromarray(np.uint8(30 + 70 * cavity)).convert("RGB")
        image.save(root / f"frame_{index:03}.jpg")
        frames.append(f"frame_{index:03}.jpg")
        if index in {2, 9}:
            name = f"mask_{index:03}.png"
            Image.fromarray(cavity.astype(np.uint8)).save(root / name)
            masks[str(index)] = name
    path = root / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "echonet",
                "videos": [
                    {
                        "video_id": "synthetic01",
                        "patient_id": "synthetic01",
                        "split": "test",
                        "frames": frames,
                        "masks": masks,
                        "ed": 2,
                        "es": 9,
                        "lvef_percent": 55.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    device = torch.device("cpu")
    with TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = synthetic_fixture(root)
        videos = load_manifest(manifest, root, "echonet", "test")
        model = make_model("echonet").eval()
        contours, function = evaluate(model, videos, "echonet", device, "fp32", 1, 3)
        assert [row["phase"] for row in contours] == ["ED", "ES"]
        assert (function[0]["ed"], function[0]["es"]) == (2, 9)
        _, gt = evaluate(None, videos, "echonet", device, "fp32", 1, 3, use_gt=True)
        assert summarize_lvef(gt)["valid_N"] == 1
        assert 0 < gt[0]["prediction"] < 100
        model.num_maskmem = 0

        def reject_memory(*_):
            raise AssertionError("Memory-free execution invoked a memory module")

        read_hook = model.memory_attention.register_forward_hook(reject_memory)
        write_hook = model.memory_encoder.register_forward_hook(reject_memory)
        try:
            with torch.inference_mode():
                outputs = list(
                    decode_video(
                        model, [torch.zeros(1, 2, 3, 128, 128)], device, "fp32"
                    )
                )
            assert len(outputs) == 2 and outputs[0].shape == (1, 2, 128, 128)
        finally:
            read_hook.remove()
            write_hook.remove()
    print(
        "PASS: manifest, chunked memory, independent-frame inference, fixed ED/ES, and GT geometry"
    )


if __name__ == "__main__":
    main()
