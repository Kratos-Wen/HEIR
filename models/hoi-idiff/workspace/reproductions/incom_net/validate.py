"""Real pretrained-feature and all-MFT-branch validation on a training image.

This is an implementation qualification, never an accuracy proxy or paper run.
Does not read test annotations or launch/reserve GPUs.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import time

import torch

from .backbones import WORKSPACE, digest, load_pretrained
from .model import InCoMConfig, InCoMHead, focal_mft_loss, human_entity_pairs, inference_scores
from .vcoco import NativeVCOCO, associate_pairs, export_vcoco


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--detector", type=Path, default=WORKSPACE.parent / "pvic/checkpoints/detr-r50-vcoco.pth")
    parser.add_argument("--clip", type=Path, default=WORKSPACE / "assets/incom_net_reproduction/ViT-L-14-336px.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, default=WORKSPACE / "reference_repos/pvic/vcoco/instances_vcoco_trainval.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if "COCO_train2014_" not in args.image.name:
        raise ValueError("Qualification must use a training image, not test selection")
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    start = time.monotonic()
    cfg = InCoMConfig()
    extractor = load_pretrained(args.detector, args.clip, args.device)
    extractor.train()
    assert not extractor.detector.training and not extractor.clip.training
    data = NativeVCOCO(args.annotations, args.image.parent,
                        WORKSPACE.parent / 'data/v-coco/data/splits/vcoco_trainval.ids', augment=False)
    index = next(i for i, row in enumerate(data.annotations) if row['file_name'] == args.image.name)
    image, target = data[index]
    image = image.to(args.device)
    target = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in target.items()}
    record = extractor([image])[0]
    pairs = human_entity_pairs(record["labels"])
    if not len(pairs):
        raise ValueError("Qualification image produced no human-entity pairs")
    head = InCoMHead(cfg).to(args.device).train()
    inputs = (record["detector_layers"], record["vlm_layers"], record["normalized_boxes"],
              record["grid"], record["cnn_tokens"], pairs)
    branches = head(*inputs)
    targets, valid = associate_pairs(record, pairs, target, data.compatibility.to(args.device))
    if targets.sum() == 0:
        raise ValueError("No positive train annotation matched: qualification needs a positive pair")
    loss, terms = focal_mft_loss(branches, targets, valid, alpha=.5, gamma=.1)
    loss.backward()
    trainable = {name: p for name, p in head.named_parameters() if p.requires_grad}
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable.values())
    assert all(not p.requires_grad and p.grad is None for p in extractor.parameters())
    head.eval()
    with torch.no_grad():
        outputs = head(*inputs)
        assert set(outputs) == {"full"}
        scores = inference_scores(outputs["full"]["logits"], record["scores"][pairs], valid)
        assert torch.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
        p, action = valid.nonzero(as_tuple=True)
        exported = export_vcoco({'boxes': record['boxes'], 'pairing': pairs[p], 'scores': scores[p, action],
                                'labels': action, 'size': record['size']}, target['image_id'],
                                (int(record['size'][1]), int(record['size'][0])))
        assert exported and all('cut_obj' in row and 'cut_instr' in row for row in exported)
    report = {
        "evidence_type": "UNOFFICIAL_PAPER_REIMPLEMENTATION",
        "qualification_only_not_accuracy": True,
        "target_kind": "actual V-COCO trainval visible-role annotations, class-aware pair matching",
        "matched_positive_edges": int(targets.sum()), "native_train_images": len(data),
        "annotation_sha256": digest(args.annotations), "exported_records": len(exported),
        "image": str(args.image), "image_sha256": digest(args.image),
        "config": asdict(cfg), "device": args.device, "seconds": time.monotonic() - start,
        "instances": len(record["labels"]), "pairs": len(pairs), "grid": record["grid"],
        "detector_layers_shape": list(record["detector_layers"].shape),
        "vlm_layers_shape": list(record["vlm_layers"].shape),
        "cnn_shape": list(record["cnn_tokens"].shape),
        "head_parameters": sum(p.numel() for p in head.parameters()),
        "loss": float(loss.detach()), "branch_losses": {k: float(v.detach()) for k, v in terms.items()},
        "all_trainable_gradients_finite": True, "frozen_backbones_unchanged": True,
        "strict_pretrained_load": True, "test_read": False,
        "detector_sha256": digest(args.detector), "clip_sha256": digest(args.clip),
        "torch_version": torch.__version__,
        "clip_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=WORKSPACE / "reference_repos/CLIP", text=True).strip(),
        "source_sha256": {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "reconstruction_spec_sha256": digest(Path(__file__).with_name("reconstruction.json")),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
