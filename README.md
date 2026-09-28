# HEIR Baseline Evaluation

This branch contains the HEIR relation and complete-set scoring tools used with baseline predictions. The CoRISP model, its training entry points and its set predictor are on `main`. V-COCO baseline implementations and adapters are on `reproductions/vcoco-baselines`.

## Contents

| Entry point | Purpose |
| --- | --- |
| `scorer/evaluate_heir_predictions.py` | Role mAP and role-free HOI mAP from baseline JSONL predictions |
| `scorer/evaluate_heir_sets.py` | Complete-set evaluation and structural breakdowns |
| `scorer/heir_eval_v02/` | Matching, AP computation and schema validation |
| `scorer/state_map.py` | Per-state MAP set decoder shared with `main` |

The repository contains source only. Images, annotations and predictions are not included; trained checkpoints are published separately and listed with their SHA-256 in `configs/checkpoints.toml`.

## Evaluate a released model

Each model has one configuration file; switching the evaluated model means switching the file.

```bash
cp configs/paths.example.toml configs/paths.toml     # dataset, support table, Python environments, assets
python evaluate.py --list
python evaluate.py configs/heir/rlipv2_swinl.toml    # results in runs/rlipv2_swinl/results.json
CUDA_VISIBLE_DEVICES=1 python evaluate.py configs/heir/gen_vlkt_l.toml --output runs/gen_vlkt_l
```

For a baseline configuration, `evaluate.py` fetches the pinned upstream source and applies the HEIR adapters, links the dataset and inference assets, downloads the checkpoint from its published link into `checkpoint_dir` when absent (requires `gdown`), verifies its SHA-256, exports test predictions and computes Role mAP, HOI mAP and Set mAP with top-k and MAP decoding. Stages whose outputs exist are reused; `--force` recomputes them and `--dry-run` prints the commands. Each model family needs its own environment with the requirements of its upstream repository; set the interpreter per family under `[python]` in `configs/paths.toml`.

| Configuration | Model |
| --- | --- |
| `configs/heir/qpic_r50.toml`, `qpic_r101.toml` | QPIC (ResNet-50, ResNet-101) on HEIR |
| `configs/heir/muren.toml` | MUREN on HEIR |
| `configs/heir/gen_vlkt_s.toml`, `gen_vlkt_l.toml` | GEN-VLKT-S, GEN-VLKT-L on HEIR |
| `configs/heir/rlipv2_swint.toml`, `rlipv2_swinl.toml` | RLIPv2 (Swin-T, Swin-L) on HEIR |
| `configs/heir/sov_stg_s.toml`, `sov_stg_l.toml`, `sov_stg_swinl.toml` | SOV-STG-S, SOV-STG-L (ResNet-101), SOV-STG (Swin-L) on HEIR |
| `configs/heir/sov_stg_vla_s.toml` | SOV-STG-VLA-S on HEIR |
| `configs/vcoco/*.toml` | QPIC, MUREN, GEN-VLKT, RLIPv2, SOV-STG-L and SOV-STG-VLA-S on V-COCO, from the authors' released checkpoints |
| `configs/heir/corisp.toml`, `configs/vcoco/corisp.toml` | CoRISP on HEIR and V-COCO, evaluated from `main` |

A V-COCO configuration checks out the pinned release of `reproductions/vcoco-baselines` into `external/`, fetches that model's upstream source, downloads the authors' checkpoint when a direct or Google Drive link exists (otherwise it names the file to place in `checkpoint_dir`), verifies its SHA-256, runs inference and reports the official role AP (Scenario 1 and 2, omitting `point`). Set `vcoco` in `configs/paths.toml` to the official V-COCO checkout. The complete-set V-COCO evaluator is `evaluation/` on that branch. Upstream code loads full training checkpoints with `torch.load`; under PyTorch 2.6 or later, run those environments with `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` and use only the checkpoints listed in `configs/checkpoints.toml`, whose SHA-256 is verified before loading.

Baseline checkpoints contain model weights, the final epoch and training arguments without local paths (`tools/strip_checkpoint.py`). The CoRISP configurations record the checkpoint, the `main` commit and its evaluation commands; `evaluate.py` does not run them.

## Baseline training recipes

| Model | Variants | Directory |
| --- | --- | --- |
| QPIC | R50, R101 | [models/qpic](models/qpic/README.md) |
| MUREN | R50 | [models/muren](models/muren/README.md) |
| SOV-STG | S, L (R101), Swin-L; SOV-STG-VLA-S | [models/sov-stg](models/sov-stg/README.md) |
| GEN-VLKT | S, L | [models/gen-vlkt](models/gen-vlkt/README.md) |
| RLIPv2 | Swin-T, Swin-L | [models/rlipv2](models/rlipv2/README.md) |

Each model directory contains the HEIR adapters as patches against pinned upstream commits (`patches/`, `sources.json`), a portable runner (`baseline.py`), the asset and command specification (`baseline.json`, `assets.example.json`), a copy of the scorer and data tools of this branch (`workspace/heir/`) and a manifest checked by `baseline.py verify`. Upstream source is not included: the `sources` command clones each repository, checks out the pinned commit, applies the patch and verifies every patched file. Every recipe then runs `prepare`, `support`, `init`, `train`, `export`, `score-role`, `score-hoi`, `score-sets-topk` and `score-sets-map` with one output directory. Official hyper-parameters are kept; runs use seed 42 and the fixed final epoch without validation or test selection. Each model directory is self-contained and has its own dependency requirements. Do not install all model families into one environment.

```bash
cd models/rlipv2
python -m unittest test_baseline.py
python baseline.py verify
python baseline.py run sources
python baseline.py link --set HEIR=/path/to/HEIR --set HEIR_SUPPORT=/path/to/support.csv \
  --set ROBERTA_BASE=/path/to/roberta-base --set RLIPV2_SWINL_PRETRAIN=/path/to/RLIP_PDA_v2_SwinL_..._checkpoint0019.pth
for phase in prepare support train export score-role score-hoi score-sets-topk score-sets-map; do
  python baseline.py run rlipv2_swinl-$phase --output ./runs/rlipv2_swinl
done
```

`prepare` writes the HICO-DET layout (437 nouns with person = 0, 225 (verb, role) interactions) with `tools/convert_heir_to_hico.py`; `support` writes the [437, 225] output-support mask from the action--noun--role support table (CSV with columns `verb`, `role`, `noun`) with `tools/build_support_mask.py`. The mask is applied where each model's official code applies its co-occurrence matrix, before its top-100 selection. Pair models with 225 interactions represent the entries whose (verb, role) occurs in training; the remainder is listed in the mask report.

## Installation and inputs

Use Python 3.11 and install the dependencies in `requirements.txt` in a dedicated environment. The HEIR root must contain its vocabulary and `annotations/train.json`, `annotations/val.json`, and `annotations/test.json`. The supplied `classes.json` must match the prediction head's object and interaction indices.

Each prediction line has an `image_id`, original-pixel `boxes` in xyxy format, `labels`, `box_scores`, and `hois`. An interaction has `subject_id`, `object_id`, `category_id`, and `score`; the endpoint indices address that line's boxes. Include one line per split image, in annotation order, including images with no predictions.

## Relation metrics

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python scorer/evaluate_heir_predictions.py \
  --heir-root /path/to/HEIR --classes /path/to/classes.json \
  --split test --predictions /path/to/test_predictions.jsonl \
  --output-dir outputs/role
python scorer/evaluate_heir_predictions.py \
  --heir-root /path/to/HEIR --classes /path/to/classes.json \
  --split test --predictions /path/to/test_predictions.jsonl \
  --hoi --output-dir outputs/hoi
```

Role AP averages supported action--noun--role classes. HOI projection removes roles and retains the maximum confidence for duplicate person--entity--action predictions. Subject and participant matches require IoU at least 0.5 and the required semantic labels. Rare, non-rare and unseen categories follow training-instance counts.

## Complete-set evaluation

Role mAP and HOI mAP above follow the conventional relation protocol and do not depend on any set decoder. Set mAP provides two decoders, top-k decoding and MAP decoding; both use the same matching and AP computation.

```bash
# top-k decoding
python scorer/evaluate_heir_sets.py \
  --heir-root /path/to/HEIR --classes /path/to/classes.json \
  --split test --predictions /path/to/test_predictions.jsonl \
  --construction top-k --max-sets 100 --output-dir outputs/sets_topk
# MAP decoding
python scorer/evaluate_heir_sets.py ... --construction map --max-sets 100 --output-dir outputs/sets_map
```

Pair-output predictions are first grouped into entity identities: same-noun boxes join a fixed representative at IoU 0.7, and duplicate edges keep their maximum confidence. At most 100 sets per image are retained; no extra relation cap is applied.

- **Top-k decoding (`top-k`).** For each person--action pair, the k highest-scoring edges form one hypothesis for every k, scored by its minimum member confidence.
- **MAP decoding (`map`).** Each candidate entity takes no role or one role, with log weights log(1 - sum p) and log p from its edge confidences, and zero count potentials. The decoder returns the best assignment of every nonempty (cardinality, role-count) state, at most eight per person--action pair, scored by its normalised probability P(S). `scorer/state_map.py` is identical to `evaluation/native.py` on `main`, so the CoRISP sets and pair-output models are decoded by the same function. Models with their own set distribution submit those hypotheses in `sets` and score them with `--construction native` under the same eight-per-pair and 100-per-image budget.

MAP decoding treats edge confidences as probabilities, whereas top-k decoding uses only their ranking. Matching uses shared image-level, noun-compatible entity identities at IoU 0.5. A true positive requires the entire participant--role set to match.

Run full-dataset scoring on an allocated compute node or a workstation, with one scoring process and bounded memory. The code checks prediction coverage and class-order compatibility before scoring. These commands perform evaluation only and do not train or select a model.

## License

Original evaluation code is provided under the [PolyForm Noncommercial License 1.0.0](LICENSE). Dataset and model assets retain their separate terms and are not distributed here.
