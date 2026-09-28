# sov-stg: HEIR

SOV-STG (S, L with ResNet-101, Swin-L) and SOV-STG-VLA-S training on HEIR.

This directory contains the HEIR dataset, export and initialisation adapters as patches against pinned upstream commits, a portable command runner and asset specifications. It does not contain upstream source code, datasets, checkpoints or prediction files. The `sources` command clones each upstream repository listed in `sources.json`, checks out the pinned commit, applies the patch from `patches/` and verifies the SHA-256 of every patched file. Install the requirements of the corresponding implementation in `workspace/reference_repos/` in a dedicated environment; the upstream licenses apply to the checked-out source and to the patched files.

Upstream repositories:

- SOV-STG: https://github.com/cjw2021/SOV-STG (commit `d7ad01f`)
- SOV-STG-VLA: https://github.com/cjw2021/SOV-STG-VLA (commit `64db7ca`)
- CLIP: https://github.com/openai/CLIP (commit `d05afc4`)

## Commands

| Command | Processes | Required assets |
| --- | ---: | --- |
| `sources` | 1 | -- |
| `sov_stg_s-prepare` | 1 | `HEIR` |
| `sov_stg_s-support` | 1 | `HEIR_SUPPORT` |
| `sov_stg_s-init` | 1 | `DN_DEFORMABLE_DETR_R50` |
| `sov_stg_s-train` | 2 | -- |
| `sov_stg_s-export` | 1 | -- |
| `sov_stg_s-score-role` | 1 | `HEIR` |
| `sov_stg_s-score-hoi` | 1 | `HEIR` |
| `sov_stg_s-score-sets-topk` | 1 | `HEIR` |
| `sov_stg_s-score-sets-map` | 1 | `HEIR` |
| `sov_stg_l-prepare` | 1 | `HEIR` |
| `sov_stg_l-support` | 1 | `HEIR_SUPPORT` |
| `sov_stg_l-init` | 1 | `DN_DAB_DEFORMABLE_DETR_R101` |
| `sov_stg_l-train` | 4 | -- |
| `sov_stg_l-export` | 1 | -- |
| `sov_stg_l-score-role` | 1 | `HEIR` |
| `sov_stg_l-score-hoi` | 1 | `HEIR` |
| `sov_stg_l-score-sets-topk` | 1 | `HEIR` |
| `sov_stg_l-score-sets-map` | 1 | `HEIR` |
| `sov_stg_swinl-prepare` | 1 | `HEIR` |
| `sov_stg_swinl-support` | 1 | `HEIR_SUPPORT` |
| `sov_stg_swinl-init` | 1 | `DN_DAB_DEFORMABLE_DETR_SWIN_L` |
| `sov_stg_swinl-train` | 2 | -- |
| `sov_stg_swinl-export` | 1 | -- |
| `sov_stg_swinl-score-role` | 1 | `HEIR` |
| `sov_stg_swinl-score-hoi` | 1 | `HEIR` |
| `sov_stg_swinl-score-sets-topk` | 1 | `HEIR` |
| `sov_stg_swinl-score-sets-map` | 1 | `HEIR` |
| `sov_stg_vla_s-prepare` | 1 | `HEIR` |
| `sov_stg_vla_s-support` | 1 | `HEIR_SUPPORT` |
| `sov_stg_vla_s-labels` | 1 | -- |
| `sov_stg_vla_s-init` | 1 | `DN_DEFORMABLE_DETR_R50` |
| `sov_stg_vla_s-train` | 2 | `CLIP_VIT_B32`, `BLIP2_PRETRAINED`, `EVA_VIT_G` |
| `sov_stg_vla_s-export` | 1 | `CLIP_VIT_B32`, `BLIP2_PRETRAINED`, `EVA_VIT_G` |
| `sov_stg_vla_s-score-role` | 1 | `HEIR` |
| `sov_stg_vla_s-score-hoi` | 1 | `HEIR` |
| `sov_stg_vla_s-score-sets-topk` | 1 | `HEIR` |
| `sov_stg_vla_s-score-sets-map` | 1 | `HEIR` |

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
| `BLIP2_PRETRAINED` | LAVIS BLIP-2 `blip2_pretrained.pth` | `f31f96e4a97c…` |
| `CLIP_VIT_B32` | OpenAI CLIP `ViT-B-32.pt` (`clip.load("ViT-B/32")` download) | `40d365715913…` |
| `DN_DAB_DEFORMABLE_DETR_R101` | SOV-STG README, DN-DAB-Deformable-DETR R101 | `6a64b7b6e215…` |
| `DN_DAB_DEFORMABLE_DETR_SWIN_L` | SOV-STG README, DN-DAB-Deformable-DETR Swin-L | `746370945683…` |
| `DN_DEFORMABLE_DETR_R50` | detrex DN-Deformable-DETR R50 (50 epochs), converted as in the SOV-STG README | `6b7ff178954e…` |
| `EVA_VIT_G` | LAVIS BLIP-2 `eva_vit_g.pth` | `99d2bb36c6b5…` |
| `HEIR` | HEIR dataset root (`vocabulary.json`, `annotations/{train,val,test}.json`, `images/`) | -- |
| `HEIR_SUPPORT` | HEIR action--noun--role support table (CSV with columns `verb`, `role`, `noun`) | -- |

Link every required asset listed for the selected command using `--set KEY=/path/to/asset`. `assets.example.json` lists all keys, and `baseline.json` specifies their destination paths. `show` prints a command without executing it. Use the same output directory for all phases of one run.

Each recipe runs the phases in this order: `sources` (once per directory), `prepare` (HEIR to HICO-DET layout), `support` (output-support mask), `init` (parameter conversion), `labels` (GEN-VLKT and SOV-STG-VLA only: triplet text labels), `train`, `export` (test predictions of the final epoch), `score-role`, `score-hoi`, `score-sets-topk` and `score-sets-map`. Official hyper-parameters are kept; the per-process batch size and gradient accumulation reproduce the official global batch on the declared world size. Runs use seed 42 and the fixed final epoch; no validation or test result is used to select a checkpoint. Distributed recipes retain their declared world size; the runner supports explicit node count, processes per node, node rank and master address. CUDA extensions must be built for the selected PyTorch/CUDA environment. Dataset and model assets require their providers' licenses.

The support mask is applied where each model's official code applies its co-occurrence matrix, before its top-100 selection. Role mAP and HOI mAP follow the conventional relation protocol. Set mAP has two decoders, described in the branch README: `score-sets-topk` (top-k decoding) and `score-sets-map` (MAP decoding, shared with the CoRISP set decoder). Use only trusted checkpoints. Run training, export and scoring on allocated compute resources rather than a cluster login node.
