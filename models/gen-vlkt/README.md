# gen-vlkt: V-COCO

GEN-VLKT released-checkpoint inference for the small and large configurations.

This directory contains the V-COCO adapters as patches against pinned upstream commits (`patches/`, `sources.json`), a portable command runner and asset specifications. It does not contain upstream source code, datasets, checkpoints or prediction caches. `python baseline.py run sources` clones each upstream repository, checks out the pinned commit, initialises its pinned submodules where listed, applies the patch and verifies every file the release was tested with. Install the requirements of the corresponding implementation in `workspace/reference_repos/` in a dedicated environment; the upstream licenses apply to the checked-out source and to the patched files.

## Commands

| Command | Processes | Required assets |
| --- | ---: | --- |
| `gen_vlkt_l-infer` | 1 | `VCOCO`, `GEN_VLKT_L_PARAM_PATH`, `GEN_VLKT_L_CLIP_MODEL` |
| `gen_vlkt_l-score` | 1 | `VCOCO` |
| `gen_vlkt_s-infer` | 1 | `VCOCO`, `GEN_VLKT_S_PARAM_PATH`, `GEN_VLKT_S_CLIP_MODEL` |
| `gen_vlkt_s-score` | 1 | `VCOCO` |

Run from this directory:

```bash
python -m unittest test_baseline.py
python baseline.py verify
python baseline.py run sources
python baseline.py commands
python baseline.py link --set VCOCO=/path/to/v-coco
python baseline.py check gen_vlkt_l-infer
python baseline.py show gen_vlkt_l-infer --output ./runs/model
python baseline.py run gen_vlkt_l-infer --output ./runs/model
```

Link every required asset listed for the selected command using `--set KEY=/path/to/asset`. `assets.example.json` lists all keys, and `baseline.json` specifies their destination paths. `show` prints a command without executing it. Use the same output directory for all phases of one run.

For training recipes, follow the listed preparation and validation phases before training, and evaluate the resulting model with the test/scoring phases. Distributed recipes retain their declared world size; the runner supports explicit node count, processes per node, node rank and master address. CUDA extensions must be built for the selected PyTorch/CUDA environment. Dataset and model assets require their providers' licenses.

Official role AP is computed separately from the common complete-set evaluator in the branch's `evaluation/` directory. Use only trusted checkpoints and caches. Run inference and scoring on allocated compute resources rather than a cluster login node.
