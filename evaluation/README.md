# V-COCO Complete-Set Evaluation

`scorer.py` implements person clustering, native role-slot assembly, matching and complete-set AP. It uses predicted-person IoU 0.7 for fixed-representative clustering and IoU 0.5 for ground-truth matching. It does not apply a confidence threshold or an additional output cap. All averages over the 21 role-bearing actions excluding `point`; Dual averages `hit`, `eat` and `cut`.

## Inputs and execution

The model-level official-scoring command produces an `official_metrics.json` ledger identifying the input cache, annotations, evaluator and their hashes. Create a manifest referring to that ledger:

```json
[
  {"model": "qpic-r50", "ledger": "/path/to/official_metrics.json"}
]
```

With NumPy and the official V-COCO evaluator's dependencies installed, run from an allocated Slurm compute step:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
srun --ntasks=1 --cpus-per-task=1 --mem=8G --time=00:30:00 \
  python evaluation/run.py --manifest /path/to/manifest.json --output outputs/sets
```

The CLI checks that it is running on the allocated compute node before loading prediction caches. Do not run full evaluation on a login node. Existing output files are not overwritten. Pickle inputs must come from a trusted source. For non-Slurm environments, `scorer.evaluate` is the pure scoring API; supply the official evaluator's image records and predictions grouped by image.

## Tests

From this directory:

```bash
python -m unittest test_scorer.py test_compute_guard.py
```

These tests use synthetic inputs and require no dataset or model weights.
