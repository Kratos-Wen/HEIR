# HEIR dataset

**Dataset:** [KratosWen/HEIR on Hugging Face](https://huggingface.co/datasets/KratosWen/HEIR)

## Release status

As verified on 2026-09-28, the Hugging Face repository is public and ungated. Its image files are accessible without an account or access token. The paper annotations, vocabulary and dataset card have not yet been published. **The current image-only repository is not a complete benchmark release.**

| Component | Public availability |
| --- | --- |
| Paper images | All 18,730 present; SHA-256 matches the paper image manifest |
| Train / validation / test annotations | Pending publication |
| Vocabulary and image provenance manifest | Pending publication |
| Dataset card and annotation license | Pending publication |

The image check used the immutable Hugging Face revision [`05ba21b`](https://huggingface.co/datasets/KratosWen/HEIR/tree/05ba21b81fa38356a2d1828a9d97ad14c94dedc2). It compared the stored file hashes with the paper's image manifest without downloading the full image corpus.

## Paper version

| Split | Images | Boxes | Relations |
| --- | ---: | ---: | ---: |
| Train | 15,158 | 62,531 | 54,277 |
| Validation | 615 | 2,809 | 2,743 |
| Test | 2,957 | 13,005 | 11,783 |
| Total | 18,730 | 78,345 | 68,803 |

The vocabulary contains 105 actions, 437 nouns and six functional roles. The paper dataset is `v1.0-rc2`; its images reuse files in `release/v1.0-rc1/images/`. That older image directory contains 19,162 files, including images outside the paper splits. Do not treat the entire image directory as the benchmark or create replacement splits.

A pinned download tool, metadata checksums and native-directory setup instructions are prepared for the complete release. They will be published after the missing dataset files become available. Image data will remain on Hugging Face; GitHub provides code, release information and download instructions.

## Formats and use

Annotations use original-image pixel coordinates and shared within-image box IDs. The `agent_only` flag limits supervision and evaluation to the annotated actor/action. See [Input Formats](DATA_FORMATS.md), [Training and Evaluation](REPRODUCTION.md) and [Assets](ASSETS.md).

Original images retain their source licenses and conditions; the repository's code license does not relicense them. Please cite the [HEIR paper](../CITATION.bib) when using the benchmark.
