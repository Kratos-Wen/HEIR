# Troubleshooting

| Symptom | Check |
| --- | --- |
| `No module named torch` | Install the paired torch/torchvision wheels before the requirements file. |
| Missing `torchvision::nms` or incompatible torchvision operators | Reinstall both packages from the same CPU or CUDA wheel index. |
| Missing `MultiScaleDeformableAttention` | Build H-DETR's extension using the active Python and matching CUDA toolkit. |
| `CUDA_HOME` or `nvcc` not found | Load the local CUDA toolkit in the allocated GPU job. |
| `Source differs from the release manifest` | Restore the release source; after an intentional code change, run `python scripts/check_release.py --update-manifests`, review the diff and rerun tests. |
| Asset checksum mismatch | Check the exact file against `configs/required_assets.json`; do not bypass the checksum or substitute another prototype order. |
| Checkpoint input or support mismatch | Use the vocabulary, training annotations, detector, prototypes and support inventory associated with that checkpoint. |
| Existing output error | Choose a new output path. Prediction and scoring entry points do not overwrite existing results. |
| Missing prediction images | Include one JSONL record per evaluation image, including images whose `sets` list is empty. |
| Unknown or duplicate entity IDs | Keep IDs image-local and reuse each physical entity's ID across all events in that image. |

## Memory and shared machines

Use `--workers 0` for HEIR initially and set OMP, MKL, OpenBLAS and NumExpr thread counts to one. On a cluster, run model loading, compilation, scoring and tests on compute nodes with explicit resources. Do not run full-dataset work on a login node.

## Getting help

Use the bug-report template when repository issues are available. Include the source revision, Python/PyTorch/torchvision/CUDA versions, a minimal command and a sanitized traceback. Do not attach credentials, private paths, dataset images, checkpoints or large prediction files. See [Security](../SECURITY.md) for sensitive reports.
