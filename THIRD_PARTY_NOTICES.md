# Third-party notices

The model code under `src/pulse/models/` derives from Meta's Segment Anything Model 2. It retains applicable source notices and the upstream [Apache 2.0 license](LICENSE-SAM2). See the [SAM2 upstream repository](https://github.com/facebookresearch/sam2) for model weight terms and documentation.

PULSE adapts the semantic decoder, class-conditioned output, memory encoding and sequence inference. This packaging consolidates identical modules and uses Python's local file API for checkpoint reads. No URL handler or native extension is needed. The archived compatibility profiles retain the task-specific forward differences.

The repository does not redistribute dataset files or trained weights. Dataset and checkpoint access remain subject to their respective terms. The upstream license file identifies the SAM2 terms; the authors should specify the distribution license for their own additions before public release.
