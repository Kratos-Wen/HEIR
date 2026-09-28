# gen-vlkt: HEIR

GEN-VLKT training on HEIR for the S (ResNet-50) and L (ResNet-101) recipes.

This directory contains the HEIR dataset, export and initialisation adapters as patches against pinned upstream commits, a portable command runner and asset specifications. It does not contain upstream source code, datasets, checkpoints or prediction files. The `sources` command clones each upstream repository listed in `sources.json`, checks out the pinned commit, applies the patch from `patches/` and verifies the SHA-256 of every patched file. Install the requirements of the corresponding implementation in `workspace/reference_repos/` in a dedicated environment; the upstream licenses apply to the checked-out source and to the patched files.

Upstream repositories:

- GEN-VLKT: https://github.com/YueLiao/gen-vlkt (commit `191fc74`)
- CLIP: https://github.com/openai/CLIP (commit `d05afc4`)

## Commands

| Command | Processes | Required assets |
| --- | ---: | --- |
| `sources` | 1 | -- |
| `gen_vlkt_s-prepare` | 1 | `HEIR` |
| `gen_vlkt_s-support` | 1 | `HEIR_SUPPORT` |
| `gen_vlkt_s-labels` | 1 | -- |
| `gen_vlkt_s-init` | 1 | `DETR_R50` |
| `gen_vlkt_s-train` | 2 | `CLIP_VIT_B32` |
| `gen_vlkt_s-export` | 1 | `CLIP_VIT_B32` |
| `gen_vlkt_s-score-role` | 1 | `HEIR` |
| `gen_vlkt_s-score-hoi` | 1 | `HEIR` |
| `gen_vlkt_s-score-sets-topk` | 1 | `HEIR` |
| `gen_vlkt_s-score-sets-map` | 1 | `HEIR` |
| `gen_vlkt_l-prepare` | 1 | `HEIR` |
| `gen_vlkt_l-support` | 1 | `HEIR_SUPPORT` |
| `gen_vlkt_l-labels` | 1 | -- |
| `gen_vlkt_l-init` | 1 | `DETR_R101` |
| `gen_vlkt_l-train` | 2 | `CLIP_VIT_B32` |
| `gen_vlkt_l-export` | 1 | `CLIP_VIT_B32` |
| `gen_vlkt_l-score-role` | 1 | `HEIR` |
| `gen_vlkt_l-score-hoi` | 1 | `HEIR` |
| `gen_vlkt_l-score-sets-topk` | 1 | `HEIR` |
| `gen_vlkt_l-score-sets-map` | 1 | `HEIR` |

Run from this directory:

```bash
python -m unittest test_baseline.py
python baseline.py verify
python baseline.py commands
python baseline.py run sources
python baseline.py link --set HEIR=/path/to/HEIR --set HEIR_SUPPORT=/path/to/support.csv
python baseline.py check sources-train
python baseline.py show sources-train --output ./runs/sources
python baseline.py run sources-prepare --output ./runs/sources
```

## Assets

| Key | Source | SHA-256 |
| --- | --- | --- |
| `CLIP_VIT_B32` | OpenAI CLIP `ViT-B-32.pt` (`clip.load("ViT-B/32")` download) | `40d365715913…` |
| `DETR_R101` | `https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth` | `2c7b67e52d2e…` |
| `DETR_R50` | `https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth` | `e632da11ec76…` |
| `HEIR` | HEIR dataset root (`vocabulary.json`, `annotations/{train,val,test}.json`, `images/`) | -- |
| `HEIR_SUPPORT` | HEIR action--noun--role support table (CSV with columns `verb`, `role`, `noun`) | -- |

Link every required asset listed for the selected command using `--set KEY=/path/to/asset`. `assets.example.json` lists all keys, and `baseline.json` specifies their destination paths. `show` prints a command without executing it. Use the same output directory for all phases of one run.

Each recipe runs the phases in this order: `sources` (once per directory), `prepare` (HEIR to HICO-DET layout), `support` (output-support mask), `init` (parameter conversion), `labels` (GEN-VLKT and SOV-STG-VLA only: triplet text labels), `train`, `export` (test predictions of the final epoch), `score-role`, `score-hoi`, `score-sets-topk` and `score-sets-map`. Official hyper-parameters are kept; the per-process batch size and gradient accumulation reproduce the official global batch on the declared world size. Runs use seed 42 and the fixed final epoch; no validation or test result is used to select a checkpoint. Distributed recipes retain their declared world size; the runner supports explicit node count, processes per node, node rank and master address. CUDA extensions must be built for the selected PyTorch/CUDA environment. Dataset and model assets require their providers' licenses.

The support mask is applied where each model's official code applies its co-occurrence matrix, before its top-100 selection. Role mAP and HOI mAP follow the conventional relation protocol. Set mAP has two decoders, described in the branch README: `score-sets-topk` (top-k decoding) and `score-sets-map` (MAP decoding, shared with the CoRISP set decoder). Use only trusted checkpoints. Run training, export and scoring on allocated compute resources rather than a cluster login node.
