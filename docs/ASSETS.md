# Assets and local configuration

The repository contains source and synthetic examples only. HEIR is hosted separately on [Hugging Face](https://huggingface.co/datasets/KratosWen/HEIR); follow the [dataset release and download guide](DATASET.md). Supply local files you are authorized to use.

## Released weights

The trained models and the files they depend on are in the [CoRISP checkpoints](https://drive.google.com/drive/folders/1yxgDVaxl0HzZ860GuX3Iry2vozDIYwuo) folder:

```bash
python -m pip install gdown
gdown --folder https://drive.google.com/drive/folders/1yxgDVaxl0HzZ860GuX3Iry2vozDIYwuo -O checkpoints
```

| File | Set as | Contents |
| --- | --- | --- |
| `corisp_heir.pth` | `CORISP_CHECKPOINT` | CoRISP on HEIR, the model in the paper's results table |
| `heir_detector.pth` | `HEIR_DETECTOR` | H-DETR Swin-L detector trained on HEIR |
| `heir_prototypes.pt` | `HEIR_PROTOTYPES` | Vocabulary-aligned semantic prototypes |
| `compatibility.json` | `HEIR_COMPATIBILITY` | Admissible action--noun--role combinations |
| `heir_support.csv` | | The same combinations as a table, used by the baseline evaluation tools |
| `corisp_vcoco.pth` | `RESUME` | CoRISP on V-COCO, the model in the paper's results table |
| `vcoco_prototypes.pt` | `VCOCO_PROTOTYPES` | V-COCO action and object prototypes |
| `vcoco_role_prototypes.pt` | `VCOCO_ROLE_PROTOTYPES` | V-COCO role prototypes |

`corisp_vcoco.pth` contains the frozen DINOv3 and DINO.txt weights, which are distributed under Meta's DINOv3 agreement; `LICENSE_DINOv3.md` in the folder is a copy. The DINOv3 files for `CORISP_WEIGHTS` are not in the folder.

## Local configuration

| Variable or argument | Required input |
| --- | --- |
| `CORISP_WEIGHTS` | Directory containing `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` and `dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth` |
| `HEIR_DATA` / `--data` | HEIR vocabulary, split annotations and images |
| `HEIR_DETECTOR` / `--detector` | HEIR H-DETR detector checkpoint |
| `HEIR_PROTOTYPES` / `--prototypes` | Vocabulary-aligned semantic prototypes |
| `HEIR_COMPATIBILITY` / `--compatibility` | Shared action--noun--role support inventory |
| `CORISP_CHECKPOINT` / `--checkpoint` | `corisp_heir.pth`, or a validation-selected checkpoint from `corisp_heir.train` |
| `HDETR_CKPT` | V-COCO H-DETR initialization (`h-defm-detr-swinL-dp0-mqs-lft-iter-2stg-vcoco.pth`) |
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
