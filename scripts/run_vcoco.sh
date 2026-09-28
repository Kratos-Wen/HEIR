#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ ! -f paths.local.sh ]] || source paths.local.sh
PYTHON="${PYTHON:-python}"
STAGE="${1:---help}"
if [[ "$STAGE" == --help || "$STAGE" == -h ]]; then
  echo 'Usage: bash scripts/run_vcoco.sh {train|cache|score|set-score|validate}'
  echo 'Local input paths and distributed training settings: docs/REPRODUCTION.md'
  exit 0
fi
export PVIC_ROOT="${PVIC_ROOT:-$ROOT/vendor/pvic}" SLHOI_ROOT="${SLHOI_ROOT:-$ROOT/vendor/slhoi}"
export PYTHONPATH="$ROOT:$ROOT/src:$ROOT/integrations:$PVIC_ROOT/h_detr/models/ops:$PVIC_ROOT/pocket:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=42 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 DETR=advanced
PROTOCOL="${VCOCO_PROTOCOL:-official_trainval_test}"
case "$PROTOCOL" in
  official_trainval_test) TRAIN=trainval; EVAL=test ;;
  official_train_val_dev) TRAIN=train; EVAL=val ;;
  *) echo 'Unsupported split protocol' >&2; exit 2 ;;
esac
if [[ "$STAGE" == set-score ]]; then
  exec "$PYTHON" scripts/evaluate_vcoco_sets.py --cache "${OUTPUT:?}/cache.pkl" \
    --coverage "$OUTPUT/coverage.json" --output "$OUTPUT/set_metrics.json" \
    --vsrl-json "${VCOCO_OFFICIAL_ROOT:?}/data/vcoco/vcoco_${EVAL}.json" \
    --coco-json "$VCOCO_OFFICIAL_ROOT/data/instances_vcoco_all_2014.json" \
    --split-ids "$VCOCO_OFFICIAL_ROOT/data/splits/vcoco_${EVAL}.ids" \
    --evaluator "$VCOCO_OFFICIAL_ROOT/vsrl_eval.py"
fi
if [[ "$STAGE" == score ]]; then
  "$PYTHON" scripts/check_vcoco_coverage.py --cache "${OUTPUT:?}/cache.pkl" \
    --coverage "$OUTPUT/coverage.json" \
    --split-ids "${VCOCO_OFFICIAL_ROOT:?}/data/splits/vcoco_${EVAL}.ids"
  exec "$PYTHON" scripts/evaluate_vcoco_official.py --cache "${OUTPUT:?}/cache.pkl" \
    --vsrl-json "${VCOCO_OFFICIAL_ROOT:?}/data/vcoco/vcoco_${EVAL}.json" \
    --coco-json "$VCOCO_OFFICIAL_ROOT/data/instances_vcoco_all_2014.json" \
    --split-ids "$VCOCO_OFFICIAL_ROOT/data/splits/vcoco_${EVAL}.ids" --split-name "$EVAL" \
    --evaluator "$VCOCO_OFFICIAL_ROOT/vsrl_eval.py" --harness-root "$PVIC_ROOT" \
    --iou-thr .5 --output-json "$OUTPUT/official_metrics.json"
fi
export CORISP_HDETR_VARIANT=hdetr_corisp_role_arity_event_field
export CORISP_DINOV3_BACKBONE="${CORISP_WEIGHTS:?}/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
export CORISP_DINOTXT_WEIGHTS="$CORISP_WEIGHTS/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"
export CORISP_DINOTXT_PROTOTYPES="${VCOCO_PROTOTYPES:?}"
export CORISP_DINOTXT_ROLE_PROTOTYPES="${VCOCO_ROLE_PROTOTYPES:?}"
export CORISP_DINO_INPUT_SIZE=448 CORISP_FIELD_STEPS=2 CORISP_ROLE_ARITY_HEADS=8 CORISP_ROLE_ARITY_RANK=64
export CORISP_ROLE_ARITY_DROPOUT=0.1 CORISP_ROLE_ARITY_MAX_LOG_RESIDUAL=4.0 CORISP_ROLE_ARITY_MAX_ENERGY=4.0
export CORISP_ROLE_ARITY_MAX_STATES=8192 CORISP_SKIP_DINO_HASH=0
WORLD=8; BATCH=8; ACCUM=2
extra=(--use-checkpoint)
case "$STAGE" in
  train) [[ -z "${RESUME:-}" ]] || extra+=(--resume "$RESUME") ;;
  validate) extra+=(--validate-only) ;;
  cache) WORLD=1; BATCH=1; ACCUM=1; extra=(--cache --resume "${RESUME:?fixed trained checkpoint}") ;;
  *) echo 'Unknown stage' >&2; exit 2 ;;
esac
args=(--backbone swin_large --drop-path-rate .5 --num-queries-one2one 900 --num-queries-one2many 1500
  --pretrained "${HDETR_CKPT:?}" --dataset vcoco --partitions "$TRAIN" "$EVAL"
  --data-root "${VCOCO_ROOT:?}" --output-dir "${OUTPUT:?}" --world-size "$WORLD" --batch-size "$BATCH"
  --grad-accum-steps "$ACCUM" --seed 42 --port "${MASTER_PORT:-29542}"
  --protocol vcoco_role_arity_event_field_matched --raw-lambda 1 --repr-dim 384 --epochs 30
  --lr-head .0001 --lr-drop 20 --lr-drop-factor .2 --weight-decay .0001 --clip-max-norm .1
  --alpha .5 --gamma .1 --box-score-thresh .05 --min-instances 3 --max-instances 15
  --vcoco-role-loss-weight 1 --vcoco-train-vsrl-json "${VCOCO_OFFICIAL_ROOT:?}/data/vcoco/vcoco_${TRAIN}.json"
  --vcoco-coco-json "$VCOCO_OFFICIAL_ROOT/data/instances_vcoco_all_2014.json"
  --vcoco-eval-split-ids "$VCOCO_OFFICIAL_ROOT/data/splits/vcoco_${EVAL}.ids" --vcoco-split-protocol "$PROTOCOL"
  --checkpoint-epochs {1..30})
"$PYTHON" -c 'from corisp_heir.environment import verify_core; verify_core()'
if [[ "$STAGE" == train ]]; then
  exec "$PYTHON" -m torch.distributed.run --nnodes=2 --nproc-per-node=4 \
    --node-rank="${NODE_RANK:?0 or 1}" --master-addr="${MASTER_ADDR:?}" --master-port="${MASTER_PORT:-29542}" \
    scripts/vcoco_entry.py "${args[@]}" "${extra[@]}"
fi
exec env -u RANK -u WORLD_SIZE -u LOCAL_RANK "$PYTHON" scripts/vcoco_entry.py "${args[@]}" "${extra[@]}"
