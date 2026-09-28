# Assets and local configuration

The repository contains source and synthetic examples only. Supply local files you are authorized to use. [Required asset checksums](../configs/required_assets.json) identify the inputs expected by the released model integration.

| Variable or argument | Required input |
| --- | --- |
| `CORISP_WEIGHTS` | Directory containing the DINOv3 backbone, DINO.txt weights and tokenizer vocabulary named in the asset manifest |
| `HEIR_DATA` / `--data` | HEIR vocabulary, split annotations and images |
| `HEIR_DETECTOR` / `--detector` | HEIR H-DETR detector checkpoint |
| `HEIR_PROTOTYPES` / `--prototypes` | Vocabulary-aligned semantic prototypes |
| `HEIR_COMPATIBILITY` / `--compatibility` | Shared action--noun--role support inventory |
| `CORISP_CHECKPOINT` / `--checkpoint` | Validation-selected checkpoint from `corisp_heir.train` |
| `HDETR_CKPT` | V-COCO H-DETR initialization |
| `VCOCO_ROOT` | V-COCO images and the dataset files expected by the bundled PViC loader |
| `VCOCO_OFFICIAL_ROOT` | Official annotations, split IDs and `vsrl_eval.py` |
| `VCOCO_PROTOTYPES`, `VCOCO_ROLE_PROTOTYPES` | V-COCO action and role prototypes |
| `OUTPUT` | Directory for V-COCO checkpoints, caches and scores |

HEIR's benchmark inventory contains 10,072 admissible action--noun--role combinations, curated through AI-assisted semantic assessment and verification by two human reviewers. The vocabulary, inventory, detector and prototypes must have matching class orders.

```text
<HEIR_DATA>/
  vocabulary.json
  annotations/
    train.json
    val.json
    test.json
  <image paths referenced by file_name>
```

Annotations use original-image pixel coordinates. [Input Formats](DATA_FORMATS.md) describes the expected keys. The set scorer needs only annotations, vocabulary and predictions; it does not load images or checkpoints.

The V-COCO wrapper optionally sources `paths.local.sh` from the repository root. That file is ignored by Git. It is a shell script, so source only a file you trust. Do not place credentials or local configuration in committed examples.

For upstream access and licensing, follow [Third-party Source](../THIRD_PARTY.md). A model's source license does not grant access to restricted weights or dataset images.
