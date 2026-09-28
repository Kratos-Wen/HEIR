"""Finite real-image validation of the declared feature adapter, not HOI AP."""

import argparse
import hashlib
import json
from pathlib import Path
import socket

from PIL import Image
import torch

from .clip_features import B16_SHA256, OFFICIAL_CLIP, load_verified_b16


ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--acknowledge-independent-features', action='store_true')
    args = parser.parse_args()
    if not args.acknowledge_independent_features:
        raise ValueError('Feature extraction choices are not verified author conventions')
    if args.output.exists():
        raise FileExistsError('Preserve completed feature qualification')
    if not torch.cuda.is_available():
        raise RuntimeError('Run on an assigned GPU')
    device = torch.device('cuda:0')
    torch.manual_seed(42)
    model, preprocess = load_verified_b16(ROOT / 'assets/group_hoi_reproduction/ViT-B-16.pt', device)
    train_artifact = ROOT / 'vcoco_eval/runs/vcoco_baselines/incom_complete_adapter_seed42_20260919_r2/train_annotations.json'
    records = json.loads(train_artifact.read_text())['annotations'][:2]
    images = []
    image_sources = []
    for row in records:
        path = ROOT.parent / 'data/v-coco/images/train2014' / row['file_name']
        with Image.open(path) as im:
            images.append(preprocess(im.convert('RGB')))
        image_sources.append({'id': row['image_id'], 'path': str(path), 'sha256': digest(path)})
    images = torch.stack(images).to(device)
    with torch.no_grad():
        tokens, pooled = model.image_features(images)
        native = model.clip.encode_image(images).float()
        torch.testing.assert_close(pooled, native, rtol=1e-5, atol=1e-5)
        texts = ['a photo of a person holding a cup', 'a photo of a person riding a bicycle']
        text, unused = model.extract_features({'text_input': texts})
        direct = model.clip.encode_text(model.tokenize(texts).to(device)).float()
        torch.testing.assert_close(text, direct, rtol=0, atol=0)
        assert unused is None and tokens.shape == (2, 197, 512) and text.shape == (2, 512)
        assert torch.isfinite(tokens).all() and torch.isfinite(text).all()
        assert (tokens[0] - tokens[1]).abs().max() > 0
        assert not any(p.requires_grad or p.grad is not None for p in model.parameters())
        model.train()
        assert not any(m.training for m in model.modules())
        assert not model.clip.visual.transformer._forward_hooks
    report = {
        'scope': 'CLIP feature adapter only; not complete GroupHOI qualification',
        'passed': True, 'full_model_qualified': False, 'training_started': False,
        'author_feature_equivalence_verified': False, 'test_labels_read': False,
        'host': socket.gethostname(), 'gpu': torch.cuda.get_device_name(device),
        'weight_sha256': B16_SHA256, 'images': image_sources,
        'image_shape': list(tokens.shape), 'text_shape': list(text.shape),
        'pooled_max_error': float((pooled - native).abs().max()),
        'cls_pooled_max_error': float((tokens[:, 0] - native).abs().max()),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
        'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in
                          sorted(OFFICIAL_CLIP.glob('*.py')) + sorted(Path(__file__).parent.glob('*.py'))},
        'choices_requiring_disclosure': ['last transformer output', 'LN and projection on every token',
                                        'include CLS', 'no feature L2 normalization',
                                        'official OpenAI image preprocessing and tokenizer'],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
