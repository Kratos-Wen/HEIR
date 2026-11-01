# Input and prediction formats

All boxes use continuous `[x1, y1, x2, y2]` pixel coordinates in the original image. Entity IDs are local to an image and are shared across that image's events. An event is keyed by its person and action, with members represented as entity--role assignments.

## HEIR annotations

`vocabulary.json` contains `verbs`, `nouns` and `roles`, each a list of objects with an `id` string. The role vocabulary is `target`, `instrument`, `support`, `source`, `destination` and `constraint`. `person` must be a noun. Preserve the vocabulary order used by the support table, detector and prototypes.

Each split JSON contains an `images` list. One image has the following form:

```json
{
  "image_id": "example",
  "file_name": "images/example.jpg",
  "width": 160,
  "height": 100,
  "boxes": [
    {"id": 0, "category": "person", "bbox": [0, 0, 50, 100]},
    {"id": 1, "category": "bread", "bbox": [70, 40, 120, 80]},
    {"id": 2, "category": "knife", "bbox": [55, 25, 100, 35]}
  ],
  "relations": [
    {"subject": 0, "object": 1, "verb": "cut", "role": "target"},
    {"subject": 0, "object": 2, "verb": "cut", "role": "instrument"}
  ]
}
```

`file_name` is relative to the data root and is used for training and image inference. The CPU set scorer does not read the image. Keep any supplied evaluation metadata unchanged; the scorer validates its supported values. Annotations and predictions are matched by `image_id`, not file order.

## HEIR predictions

`corisp_heir.predict` writes one JSONL record per image, including empty outputs. The fields needed by the set scorer are:

```json
{
  "image_id": "example",
  "entities": [
    {"id": 0, "noun": "person", "box": [0, 0, 50, 100], "score": 0.99},
    {"id": 1, "noun": "bread", "box": [70, 40, 120, 80], "score": 0.95},
    {"id": 2, "noun": "knife", "box": [55, 25, 100, 35], "score": 0.92}
  ],
  "sets": [
    {
      "subject_id": 0,
      "action": "cut",
      "members": [
        {"entity_id": 1, "role": "target"},
        {"entity_id": 2, "role": "instrument"}
      ],
      "score": 0.85
    }
  ]
}
```

Every referenced entity must be present. Scores are finite probabilities in `[0, 1]`. Each set has at least one member and no duplicate member; the subject cannot also be a member under the same entity ID. Keep at most 100 sets per image. The CoRISP predictor uses at most eight count-state winners per person--action event before applying that image budget.

The predictor additionally emits `size` as `[height, width]`, `pairs` as `[subject_id, entity_id]` pairs, dense `role_scores`, `actions`, `roles`, and each set's `log_probability`. Relation scoring uses `role_scores` indexed by pair, action and role. Set scoring consumes `sets` directly; it does not threshold relation scores or rebuild their assignments.

## Potential interface

`python -m evaluation.native` accepts an NPZ file with `weights` and `energy`. `weights` has shape `[M, 1 + R]`: column zero is unselected, followed by the `R` active roles. Disallowed positive states have log weight `-inf`. `energy` has shape `[M + 1, 3**R]`, combining the total-count and role-multiplicity potentials. A role count is encoded as 0, 1 or 2+, with role zero the least-significant base-3 digit. An optional scalar `log_partition` is checked against the exact partition of the supplied potentials.

The asset-free [example](../examples/event_sets.py) exercises this interface and the set scorer together:

```bash
python -m examples.event_sets
```

## V-COCO

V-COCO uses the official detection-list cache and native role slots. `scripts/run_vcoco.sh cache` produces `cache.pkl` and `coverage.json`; `score` runs official role AP and `set-score` runs complete-slot AP. Both require the official annotations and split IDs. See [Training and Evaluation](REPRODUCTION.md) for configuration.

Pickle and PyTorch checkpoint deserialization can execute code. Load only trusted caches and checkpoints; do not treat JSON schema validation as a sandbox for them.
