# Data preparation

Acquire [EchoNet-Dynamic](https://echonet.github.io/dynamic/) and [CAMUS](https://www.creatis.insa-lyon.fr/Challenge/camus/) from their providers. The smoke check generates its inputs locally. The private A4C cohort and its annotations require authorized access. All examples below use invented identifiers.

## Common prepared data

Store a video's JPEG frames in their actual chronological order, numbered from zero. Store sparse masks as indexed PNGs using the corresponding frame numbers. Mask values are integer label IDs with background at 0. Preserve the original index mapping when extracting a subclip. For new stage 2 training, export full videos so that the sampler can extend short ED–ES intervals into available context. To evaluate archived EchoNet weights, use the benchmark JPEGs and original benchmark PNG masks. A different trace rasterization or JPEG preprocessing can change the reported scores.

The manifest can be generated from a CSV with these columns.

| Dataset | Required columns |
| --- | --- |
| All | `video_id,patient_id,split,frames_dir,masks_dir` |
| EchoNet | `ed,es,lvef_percent` |
| CAMUS | `ed,es,view,spacing_y_mm,spacing_x_mm` |

`frames_dir` and `masks_dir` are relative to the same data root. Each CSV row describes one video. ED and ES are zero-based indices within that video's exported sequence. Both must identify corresponding PNG masks. Each EchoNet LVEF value is a clinical reference in percent. Supply the provider split for EchoNet. For CAMUS, each patient contributes one 2CH and one 4CH view and must belong to one split. Spacing is in millimeters **on the 256 × 256 evaluation grid**, in row/column order. Scale native y and x spacing by native height/256 and width/256, respectively.

```bash
python tools/build_manifest.py --dataset echonet --cohort-csv data/echo_cases.csv \
  --data-root data/echonet --output data/echonet.json
```

The manifest contains a top-level dataset name and schema version, and rows like this:

```json
{
  "schema_version": 1,
  "dataset": "echonet",
  "videos": [
    {
      "video_id": "example001",
      "patient_id": "example001",
      "split": "test",
      "frames": ["example001/frames/00000.jpg", "example001/frames/00001.jpg"],
      "masks": {"0": "example001/masks/00000.png", "1": "example001/masks/00001.png"},
      "ed": 0,
      "es": 1,
      "lvef_percent": 56.0
    }
  ]
}
```

For split audits, include train, validation and test identities in **one** manifest. For private data, use one consistent pseudonymous patient key for all videos from the same patient. The loader rejects a patient key appearing across splits, as well as duplicate video IDs, missing files and invalid label indices. This checks the supplied mapping and depends on it faithfully identifying patients.

## Evaluation cohorts

- **EchoNet-Dynamic** uses the 1,275 test videos with valid annotated endpoints and a clinical LVEF reference. Preserve archived eligibility decisions and source indices when comparing with the paper. For paper mode, use the original EchoNet FileList name without its extension as the video ID. A cohort checksum detects missing, added or substituted videos. The repository does not encode a replacement phase-selection rule.
- **CAMUS** uses 50 held-out patients and 100 views for the paper's test results. Retain the benchmark sequence direction and canonical mask mapping. Use disjoint identities for any new validation and test split. The historical CAMUS checkpoint has incomplete evidence for the separation of its original validation and model-selection data. The registry identifies the test evaluation asset, not a newly verified training split.
- **Private A4C** uses 80 test videos with 324 annotated frames. A data holder can use the same manifest and metrics with the authorized cohort. The repository cannot reproduce the private score in the absence of the cohort.
