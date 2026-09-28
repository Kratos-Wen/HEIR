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

## Paper checkpoint bundle

The [paper asset folder](https://drive.google.com/drive/folders/1HDrQqFihSU33Wr56KfxksulvA7dLaZEB) contains HEIR epoch 11 (selected by validation Role mAP), V-COCO fixed final epoch 30, the HEIR detector, HEIR/V-COCO prototypes and the shared HEIR inventory. Access currently requires permission; public sharing is pending author approval.

[Released asset checksums and individual links](../configs/released_assets.json) bind this code to the exported files. These inference exports preserve every model/prototype tensor bit-for-bit, remove optimizer/RNG state and local metadata paths, and rename prototype schema identifiers to CoRISP. The support contains the same 10,072 combinations. Different serialization changes file checksums; use the matching files instead of disabling checksum checks. These files cannot resume training exactly.

Set `CORISP_CHECKPOINT` to `CoRISP_HEIR_epoch011.pth` for HEIR or pass `RESUME=.../CoRISP_VCOCO_epoch030.pth` for V-COCO caching. Set the prototype, detector and inventory variables from the table above to the corresponding downloaded files. Upstream DINOv3/DINO.txt weights, V-COCO H-DETR initialization and dataset files remain external inputs.

Validation of this export: all model tensors were compared with their original experiment checkpoints; both publication architectures loaded the exports with strict state-key checks. Full image inference and benchmark scoring were not rerun for this export. The V-COCO source run is the one underlying the paper's 73.72/76.23 official role AP; those are historical results, not a new evaluation.
