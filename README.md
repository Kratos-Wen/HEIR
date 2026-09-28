# V-COCO Baseline Reproductions

This branch contains baseline source implementations, dataset adapters and the common V-COCO complete-set evaluator. The CoRISP implementation is on `main`; HEIR scoring tools are on `reproductions/heir-baselines`.

## Models

| Model | Source and entry point |
| --- | --- |
| QPIC | [Released-checkpoint inference](models/qpic/README.md) |
| MUREN | [Released-checkpoint inference](models/muren/README.md) |
| SOV-STG | [Released-checkpoint inference](models/sov-stg/README.md) |
| GEN-VLKT | [Released-checkpoint inference](models/gen-vlkt/README.md) |
| RLIPv2 | [Released-checkpoint inference](models/rlipv2/README.md) |
| PViC | [Official implementation and training adapter](models/pvic/README.md) |
| GroupHOI | [Official core and V-COCO adapter](models/grouphoi/README.md) |
| InCoM-Net | [Independent implementation and V-COCO adapter](models/incom-net/README.md) |
| SL-HOI | [Official core and V-COCO adapter](models/sl-hoi/README.md) |
| HOI-IDiff | [Source and cache-scoring interface](models/hoi-idiff/README.md) |
| UniHOI | [Source and cache-scoring interface](models/unihoi/README.md) |

The last two entries provide source and interfaces, not a validated training recipe or a claim of reproduced paper accuracy. Each model directory is self-contained and has its own dependency requirements. Do not install all model families into one environment.

## Run a model

From a model directory, inspect its README and required assets, then use its runner:

```bash
cd models/qpic
python -m unittest test_baseline.py
python baseline.py verify
python baseline.py run sources
python baseline.py commands
python baseline.py link --set VCOCO=/path/to/v-coco
python baseline.py link --set QPIC_R50_PARAM_PATH=/path/to/qpic_resnet50_vcoco.pth
python baseline.py check qpic_r50-infer
python baseline.py run qpic_r50-infer --output ./runs/qpic-r50
python baseline.py run qpic_r50-score --output ./runs/qpic-r50
```

Upstream source code is not included: `python baseline.py run sources` clones each upstream repository at the commit pinned in `sources.json`, applies the V-COCO adapters in `patches/` and verifies every file the release was tested with. Images, annotations, weights, prediction caches and external model assets are supplied separately. `assets.example.json` lists the asset keys; `baseline.json` specifies their relative mount points and model commands. Training commands retain their specified global world size. Use `--nnodes`, `--nproc-per-node`, `--node-rank` and `--master-addr` when launching a distributed recipe across nodes.

## Complete-set evaluation

The common evaluator is in [evaluation/](evaluation/README.md). It scores complete native role-slot assignments under Scenarios 1 and 2 on all 4,946 test images. Predicted person boxes use fixed-representative clustering at IoU 0.7; matching to ground truth uses IoU 0.5. Set confidence is the minimum role-slot confidence. The evaluator averages AP over 21 role-bearing actions excluding `point`; Dual averages `hit`, `eat` and `cut`.

Official role AP is produced by each model's official-scoring command and remains a separate metric. Run full-dataset inference and scoring on a workstation or an allocated compute node, not a cluster login node. Only load trusted pickle files.

## Licenses

Third-party sources retain their included copyright notices and licenses. Original reproduction adapters and evaluation tools are covered by the [PolyForm Noncommercial License 1.0.0](LICENSE). This license does not replace or expand third-party permissions; model weights and datasets retain their providers' terms. See [NOTICE](NOTICE).
