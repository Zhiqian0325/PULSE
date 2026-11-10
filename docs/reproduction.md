# Reproducing the PULSE evaluations

The main path uses one PULSE model implementation. The encoder, transformer and memory attention are shared by the archive compatibility profiles. The retained legacy decoder and memory-encoding behavior preserve the state-dict and forward semantics of the old CAMUS and private A4C checkpoints. These profiles are used only when requested for historical evaluation. Existing model parameters and predictions are not transferred between datasets.

## Results and assets

| Paper result | Checkpoint key | Mean Dice | LVEF MAE (pp) |
| --- | --- | ---: | ---: |
| EchoNet memory, segmentation | `echonet-memory-stage1` | 92.72% | 6.29 |
| EchoNet memory, refined, LVEF-selected | `echonet-memory-stage2-lvef` | 92.75% | 4.65 |
| EchoNet memory, refined, Dice-selected | `echonet-memory-stage2-dice` | 92.84% | 4.84 |
| EchoNet memory-free, segmentation | `echonet-memory-free-stage1` | 92.63% | 6.65 |
| EchoNet memory-free, refined | `echonet-memory-free-stage2` | 92.59% | 6.27 |
| CAMUS segmentation | `camus` | 91.97% | — |
| Private A4C segmentation | `private-a4c` | 92.54% | — |

These are archived manuscript references. See [the registry](../src/pulse/paper_checkpoints.json) for checksum and cohort checks. Trained checkpoints, frozen benchmark manifests and benchmark data are external inputs to the reproduction commands. No public download is declared for the trained weights. An author preparing a submission should supply the cleared weights or an access route if claiming that external reviewers can reproduce the numbers.

For generic initialization, obtain `sam2_hiera_base_plus.pt` from the [SAM2 project](https://github.com/facebookresearch/sam2). The loader records matching, adapted, skipped and remaining randomly initialized keys. Paper evaluation uses strict loading without initialization-time adaptation.

## Evaluation choices

- EchoNet segmentation and LVEF use the dataset's original ED/ES labels. Decode all intervening frames in chronological order and keep memory across encoder chunks. Use soft LV maps for the refinement loss and hard LV masks for evaluation. Segmentation-only training does not use LVEF supervision.
- The disk calculation estimates the long axis from the mask-weighted second moments. The axis is detached during training. For near-isotropic masks the algorithm uses the vertical axis. The 128-grid mask is resampled onto a 192-grid and integrated as circular disks. Volume is a pixel-space proxy in px³ and the ratio is dimensionless. Negative LVEF estimates are preserved and invalid pairs are counted.
- Dice and boundary metrics are scored at annotated frames only. CAMUS aggregates the two views by patient. EchoNet and private A4C aggregate by video. HD95 and ASD use bidirectional surface distances. A one-empty contour is tracked explicitly. The historical private PULSE convention uses a grid-diagonal penalty for that case. Intermediate-frame predictions provide temporal context and are not densely annotated ground truth.
- The frozen EchoNet memory evaluations used CUDA BF16, same-length batches of up to four videos, and an encoder budget of 32 frames per batch. Private A4C used FP32 with eight-frame encoder chunks and continuous decoding. CAMUS used the archived predictor and canonical benchmark preprocessing. A different device, library build, grouping or preprocessing can change numerical results.

## Training choices

The portable training recipe uses the same architecture for LV, three-class and four-class tasks, with a separate checkpoint per dataset. It uses a shared transform per clip, CE plus squared-denominator Dice and a signed-distance boundary term. It has no flip, spatial scale jitter or cutout. The archived training used earlier augmentation, distributed scheduling and cohort configurations. The archived EchoNet memory stage 1 checkpoint predates the later random-anchor sampler. Its earlier archive used up-to-eight-frame tiled clips. The portable command does not recreate every archived run's update schedule or RNG trajectory.

For stage 2, short annotated intervals retain all their frames and can extend to 20 using available context. Long intervals are uniformly sampled to 20 while retaining the two annotations. Source videos shorter than 20 use all real frames. The loss operates on the predicted ED and ES contours in the same clip. Memory is bounded to seven recent entries; the stage 2 computation graph spans the clip. This is separate from native-frame test inference.

Comparative baseline implementations and their prompt registries are separate experiment assets. Point-prompted results use a positive point generated from the annotated target, shared by models under that evaluation protocol. The primary method requires no per-class spatial prompt. Published pairwise statistics use patients as the unit for CAMUS and videos for the other cohorts. This release contains the PULSE evaluator; baseline numbers cannot be regenerated from it alone.
