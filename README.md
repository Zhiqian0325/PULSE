# PULSE

PULSE is a SAM2 adaptation for prompt-free semantic segmentation of echocardiographic video. It predicts the anatomical classes jointly and propagates information through a bounded video memory. The optional second training stage uses segmentation-derived left ventricular ejection fraction (LVEF) supervision.

One model implementation, loader, training command and evaluation command serve all datasets. Dataset configuration defines the input size, foreground classes and evaluation units. The archived CAMUS and private A4C weights have compatibility profiles for their historical decoder and memory behavior. New training uses the same PULSE model across datasets.

| Dataset | Foreground classes | Evaluation grid | Boundary units |
| --- | --- | --- | --- |
| EchoNet-Dynamic | LV | 128 × 128 | px |
| CAMUS | LV, MYO, LA | 256 × 256 | mm |
| Private A4C | LV, RV, RA, LA | 256 × 256 | px |

## Install and run the CPU check

Create a Python 3.11 environment and install PyTorch for your CPU or CUDA platform. The CPU commands below use the versions in [constraints-smoke.txt](constraints-smoke.txt). Use the matching CUDA build of PyTorch and TorchVision for GPU training.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c constraints-smoke.txt .
python tools/smoke.py
python -m unittest discover -s tests
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. The smoke check generates its own images and verifies the model, FIFO across encoder chunks, fixed ED/ES geometry and the memory-free inference path. It needs no datasets or downloaded weights. Install the `train` extra for the training recipe.

## Evaluate

Obtain the public datasets from their providers and prepare the images and indexed masks described in [the data guide](docs/data.md). The guide includes the manifest schema and CSV conversion command. Paths in the manifest are relative to one explicit data root.

```bash
pulse audit-data --dataset echonet --manifest data/echonet.json --data-root data/echonet --check-masks
pulse evaluate --dataset echonet --manifest data/echonet.json --data-root data/echonet \
  --checkpoint weights/echonet-memory-stage2-lvef.pt \
  --paper-checkpoint echonet-memory-stage2-lvef --device cuda --precision bf16 \
  --output runs/echonet-memory-stage2
```

The paper checkpoint registry records the expected SHA-256, selection rule, epoch when known and cohort size. The CLI checks the checksum and loads all model tensors strictly. The model sees the complete chronological interval delimited by the dataset's annotated ED and ES. LVEF uses the predictions at those two frame indices. The order of the two annotations may differ from chronological order. Prediction-area extrema do not enter evaluation.

To compute the GT geometry reference, use the same command with `--gt-geometry` instead of the checkpoint and paper-checkpoint options. This uses the annotated endpoint contours and the same disk-volume algorithm.

The output contains a run summary, frame/class contour scores and per-video functional measurements for EchoNet. Dice is reported as a fraction in machine outputs. LVEF errors are in percentage points. Invalid geometry is counted separately and makes the full-cohort LVEF score undefined.

## Train a new model

```bash
python -m pip install -c constraints-smoke.txt '.[train]'
pulse train --dataset echonet --manifest data/echonet.json --data-root data/echonet \
  --stage 1 --initial weights/sam2_hiera_base_plus.pt \
  --device cuda --precision bf16 --batch-size 2 --output runs/stage1
pulse train --dataset echonet --manifest data/echonet.json --data-root data/echonet \
  --stage 2 --initial runs/stage1/best_dice.pt \
  --device cuda --precision bf16 --batch-size 2 --output runs/stage2
```

Stage 1 uses individual contiguous clips of at most eight frames. Stage 2 uses one sequence containing both endpoints, with a 20-frame target length. Train and validation are loaded separately and model selection uses validation only. Stage 2 is defined for EchoNet. The `--no-memory` switch trains a separate EchoNet model without memory reads or writes.

The training command is a portable recipe for new runs. The frozen paper weights came from archived training schedules and data preparation. See [the reproduction guide](docs/reproduction.md) for the exact distinction and reference numbers.
