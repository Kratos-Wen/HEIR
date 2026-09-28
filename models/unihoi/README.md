# unihoi: V-COCO

UniHOI source and a cache-scoring interface. The upstream repository (https://github.com/PanqiYang1/UniHOI_AAAI2026) ships no license file; obtain and use its code under its authors' terms. A validated training recipe is not provided.

This directory uses the official implementation at the upstream commit pinned in `sources.json`, unchanged, with a portable command runner and asset specifications. It does not contain upstream source code, datasets, checkpoints or prediction caches. `python baseline.py run sources` clones the official repository, checks out the pinned commit and verifies its files. Install the official requirements of that implementation in a dedicated environment; the official code expects the library versions listed by its authors.

## Commands

| Command | Processes | Required assets |
| --- | ---: | --- |
| `score-cache` | 1 | `VCOCO` |

Run from this directory:

```bash
python -m unittest test_baseline.py
python baseline.py verify
python baseline.py run sources
python baseline.py commands
python baseline.py link --set VCOCO=/path/to/v-coco
python baseline.py check score-cache
python baseline.py show score-cache --output ./runs/model
python baseline.py run score-cache --output ./runs/model
```

Link every required asset listed for the selected command using `--set KEY=/path/to/asset`. `assets.example.json` lists all keys, and `baseline.json` specifies their destination paths. `show` prints a command without executing it. Use the same output directory for all phases of one run.

For training recipes, follow the listed preparation and validation phases before training, and evaluate the resulting model with the test/scoring phases. Distributed recipes retain their declared world size; the runner supports explicit node count, processes per node, node rank and master address. CUDA extensions must be built for the selected PyTorch/CUDA environment. Dataset and model assets require their providers' licenses.

Official role AP is computed separately from the common complete-set evaluator in the branch's `evaluation/` directory. Use only trusted checkpoints and caches. Run inference and scoring on allocated compute resources rather than a cluster login node.
