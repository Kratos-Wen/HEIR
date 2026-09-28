# Running CoRISP

Run all commands from the repository root after [installation](INSTALLATION.md). This package supplies code only. Provide the local [assets](ASSETS.md) with the class order and configuration expected by the model. Required pretrained file names and checksums are in `configs/required_assets.json`.

## HEIR

The data directory contains `vocabulary.json`, `annotations/train.json`, `annotations/val.json`, `annotations/test.json`, and the image paths referenced by the annotations. The support JSON must use the same action, noun and role IDs. The detector checkpoint and semantic prototypes must agree with this vocabulary.

Set the following local paths:

```bash
export CORISP_WEIGHTS=/path/to/dino_weights
export HEIR_DATA=/path/to/heir
export HEIR_DETECTOR=/path/to/detector.pth
export HEIR_PROTOTYPES=/path/to/prototypes.pt
export HEIR_COMPATIBILITY=/path/to/compatibility.json
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

Train with one image per rank and two accumulated microbatches:

```bash
torchrun --standalone --nproc-per-node=4 -m corisp_heir.train \
  --data "$HEIR_DATA" --detector "$HEIR_DETECTOR" \
  --prototypes "$HEIR_PROTOTYPES" --compatibility "$HEIR_COMPATIBILITY" \
  --output outputs/heir --epochs 30 --workers 0
```

The global batch is twice the world size. `--compile-dp` compiles the same exact recurrence. `--ablation no_arity` removes G; `--ablation no_relations` removes contextual relation messages; `--ablation no_role_feedback` removes recurrent role content and uses interaction-only message weights. The final role classifier remains active. These controls preserve the set-decoding rule. Resume with `--resume` and the same configuration, inputs and world size.

The trainer saves one checkpoint per epoch. Export validation predictions with `corisp_heir.predict --split val` and evaluate them with the benchmark's relation evaluator; the training entry point does not automatically rank checkpoints. Select one checkpoint per trained model by validation Role mAP. Set its path and use it for every test metric and visualization:

```bash
export CORISP_CHECKPOINT=/path/to/validation_selected_model.pth
python -m corisp_heir.predict \
  --data "$HEIR_DATA" --split test --detector "$HEIR_DETECTOR" \
  --prototypes "$HEIR_PROTOTYPES" --compatibility "$HEIR_COMPATIBILITY" \
  --checkpoint "$CORISP_CHECKPOINT" --output outputs/heir/test.jsonl
```

Set prediction uses K=8 and a 100-set image budget. Relation marginals are exported alongside the sets and are not truncated before set prediction.

The predictor accepts checkpoints saved by the supplied trainer and validates their support and input checksums. Each JSONL record contains original-resolution `entities`, shared image-local IDs, native `sets`, and dense `role_scores` indexed by `pairs`, `actions`, and `roles`. Only load trusted checkpoints and prediction caches: PyTorch and pickle inputs can execute code during deserialization.

Score the submitted sets on CPU:

```bash
python -m evaluation.heir_sets \
  --annotations "$HEIR_DATA/annotations/test.json" \
  --vocabulary "$HEIR_DATA/vocabulary.json" \
  --predictions outputs/heir/test.jsonl --output outputs/heir/set_metrics.json
```

The scorer evaluates the submitted sets, requiring complete image coverage and enforcing the 100-set budget, declared annotation scope, shared image-level matching, and per-action AP with grouped score ties. Annotation scope is applied during scoring, not prediction. Role and HOI benchmark scoring use the exported marginal arrays and the benchmark's relation evaluator.

## V-COCO

Set `CORISP_WEIGHTS`, `HDETR_CKPT`, `VCOCO_ROOT`, `VCOCO_OFFICIAL_ROOT`, `VCOCO_PROTOTYPES`, `VCOCO_ROLE_PROTOTYPES`, and `OUTPUT` to local paths. The official evaluator and annotations are external inputs. Obtain them through the dataset's official distribution.

```bash
bash scripts/run_vcoco.sh train
RESUME=/path/to/checkpoint.pth bash scripts/run_vcoco.sh cache
bash scripts/run_vcoco.sh score
bash scripts/run_vcoco.sh set-score
```

Training uses two nodes with four GPUs each and two-step gradient accumulation. Set `NODE_RANK` and `MASTER_ADDR` on each node. Cache generation runs on one GPU; scoring uses CPU. The set scorer clusters predicted person boxes at IoU 0.7, forms one hypothesis per person and action from the highest-scoring native slots, and uses the minimum slot confidence. Matching uses IoU 0.5 and the S1/S2 missing-filler rules.

## Numerical checks

```bash
python -m pytest -q
python -m examples.event_sets
python scripts/check_release.py
python -m evaluation.native --help
python -m corisp_heir.predict --help
```

The tests use synthetic inputs and require no dataset or model weights. They check partitions and gradients, saturation at 2+, role-context and feedback operations, localization ambiguity, native state winners and complete-slot scoring. They do not substitute for an end-to-end run with user-supplied assets.
