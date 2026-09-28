# HEIR dataset

The dataset is hosted at [KratosWen/HEIR on Hugging Face](https://huggingface.co/datasets/KratosWen/HEIR). GitHub contains the download tool, pinned release metadata and format documentation; image data is stored on Hugging Face.

**Status:** images, all three annotation splits, vocabulary, provenance and checksums are public and ungated. The new annotations and vocabulary use CC BY 4.0.

## Paper version

| Split | Images | Boxes | Relations |
| --- | ---: | ---: | ---: |
| Train | 15,158 | 62,531 | 54,277 |
| Validation | 615 | 2,809 | 2,743 |
| Test | 2,957 | 13,005 | 11,783 |
| Total | 18,730 | 78,345 | 68,803 |

The vocabulary contains 105 actions, 437 nouns and six functional roles. **v1.0** contains all 18,730 paper images in `release/v1.0/images/`, alongside annotations and metadata. The downloader verifies image hashes and installs the exact paths referenced by the annotations.

## Direct annotation downloads

The download script below also installs these files and checks their hashes. For manual access:

- [Train annotations](https://huggingface.co/datasets/KratosWen/HEIR/resolve/8bcdf5ee6d545930d2574707a8f146e2c67069d9/release/v1.0/annotations/train.json)
- [Validation annotations](https://huggingface.co/datasets/KratosWen/HEIR/resolve/8bcdf5ee6d545930d2574707a8f146e2c67069d9/release/v1.0/annotations/val.json)
- [Test annotations](https://huggingface.co/datasets/KratosWen/HEIR/resolve/8bcdf5ee6d545930d2574707a8f146e2c67069d9/release/v1.0/annotations/test.json)
- [Vocabulary](https://huggingface.co/datasets/KratosWen/HEIR/resolve/8bcdf5ee6d545930d2574707a8f146e2c67069d9/release/v1.0/vocabulary.json)
- [Image provenance](https://huggingface.co/datasets/KratosWen/HEIR/resolve/8bcdf5ee6d545930d2574707a8f146e2c67069d9/release/v1.0/images.json)

## Download

Use Python 3.11 from the repository root. No Hugging Face account, token or additional Python package is required. Image content totals approximately 4.40 GB, plus annotations and metadata. On a cluster, run the full download and verification on an allocated compute node.

```bash
python scripts/download_heir.py --output data/HEIR
export HEIR_DATA="$PWD/data/HEIR"
```

For annotation-only work, including evaluation of existing predictions:

```bash
python scripts/download_heir.py --output data/HEIR --annotations-only
```

To verify an existing full download without network access:

```bash
python scripts/download_heir.py --output data/HEIR --verify-only
```

The downloader uses the exact Hugging Face commits and metadata checksums recorded in [configs/heir_dataset.json](../configs/heir_dataset.json). It downloads one file at a time, checks size and SHA-256, and reuses verified local files. A mismatch stops the download and preserves existing data. After an interrupted run, remove only the named `.part` file before retrying.

```text
HEIR/
  vocabulary.json
  annotations/
    train.json
    val.json
    test.json
  images/
    cv_<24-hex-id>.jpg
  images.json
  image_files.json
  MANIFEST.sha256
```

`images.json` records image provenance, split membership and SHA-256. `image_files.json` maps local filenames to their v1.0 Hugging Face image paths. The raw annotation JSON is nested under an `images` key; the provided downloader prepares the native format used by CoRISP.

## Evaluation scope

Annotations use pixel coordinates in the stored images. Participants share box identities within an image. The `agent_only` flag restricts supervision and evaluation to the annotated actor/action; unannotated actor/action combinations are not negative labels. See [Input Formats](DATA_FORMATS.md) and [Training and Evaluation](REPRODUCTION.md) for the implemented protocol.

Dataset annotations and images do not include trained checkpoints, detectors, semantic prototypes or model support assets. These remain separate inputs described in [Assets](ASSETS.md).

## Terms and citation

HEIR’s new annotations and vocabulary are released under [CC BY 4.0](https://huggingface.co/datasets/KratosWen/HEIR/blob/main/LICENSE.md). Original images remain subject to their source datasets' licenses and conditions; the repository's code license does not relicense those images. Source identifiers are retained in `images.json`.

Please cite the [HEIR paper](../CITATION.bib) when using the benchmark.
